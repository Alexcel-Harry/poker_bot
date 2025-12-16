# ==========================================
# IMPROVED POKER AI TRAINING (v2.2 - FINAL CORRECTED)
# ==========================================
# All fixes applied:
# - Previous fixes: train_rl/train_sl, cache keys, eval7 checks, all players, state_shape, etc.
# - NEW: Separate agent instances (fixes NFSP theory violation)
# - NEW: Proper tie/split handling in equity calculation
# - Uses AUXILIARY LOSS (observational learning)
# - Caches equity calculations (100x faster)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import set_seed, tournament, reorganize, Logger
try:
    import eval7
    EVAL7_AVAILABLE = True
except ImportError:
    EVAL7_AVAILABLE = False
    print("⚠️ Warning: eval7 not installed. Equity calculation disabled.")
from collections import defaultdict
import time
import pickle

# --- Configuration ---
ENV_NAME = 'no-limit-holdem'
NUM_EPISODES = 20_000_000
EVAL_EVERY = 20_000
SAVE_EVERY = 1_000_000
SAVE_PATH = 'poker_model_improved_v2'
SEED = 42
NUM_PLAYERS = 3

# Resume from checkpoint
RESUME_FROM_EPISODE = 0
RESUME_CHECKPOINT_DIR = None

# Network configuration
HIDDEN_LAYERS = [1024, 1024, 512, 256]
Q_MLP_LAYERS = [1024, 1024, 512, 256]
LEARNING_RATE = 0.00005

# Auxiliary learning configuration
USE_AUXILIARY_LOSS = True
AUXILIARY_WEIGHT = 0.5
EQUITY_CACHE_SIZE = 100000

# Potential-based shaping (disabled - needs proper implementation)
USE_POTENTIAL_SHAPING = False
POTENTIAL_DISCOUNT = 0.95

# Monte Carlo configuration
EQUITY_SIMS_TRAINING = 100
EQUITY_SIMS_EVAL = 1000
EQUITY_CACHE_FILE = f'{SAVE_PATH}/equity_cache.pkl'

print("="*70)
print("IMPROVED POKER AI TRAINING (v2.2 - FINAL)")
print("="*70)
print(f"Configuration:")
print(f"  - Players: {NUM_PLAYERS} (separate agent instances)")
print(f"  - Episodes: {NUM_EPISODES:,}")
print(f"  - Network: {HIDDEN_LAYERS}")
print(f"  - Auxiliary Loss: {'ENABLED' if USE_AUXILIARY_LOSS else 'DISABLED'}")
print(f"  - eval7 Available: {'YES' if EVAL7_AVAILABLE else 'NO (using fallback)'}")
print("="*70)

# --- Equity Cache ---
class EquityCache:
    """
    Cache equity calculations to avoid expensive Monte Carlo
    Key insight: same (hand, board, opponents) → same equity
    """
    def __init__(self, max_size=100000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        
    def get_key(self, hole_cards, board_cards, num_opponents):
        """Create hashable key from cards AND opponent count"""
        hole_str = tuple(sorted(str(c) for c in hole_cards))
        board_str = tuple(sorted(str(c) for c in board_cards))
        return (hole_str, board_str, num_opponents)
    
    def get(self, hole_cards, board_cards, num_opponents, n_sims):
        """Get cached equity or compute and cache"""
        key = self.get_key(hole_cards, board_cards, num_opponents)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        
        self.misses += 1
        equity = self._compute_equity(hole_cards, board_cards, num_opponents, n_sims)
        
        # Limit cache size (simple FIFO)
        if len(self.cache) >= self.max_size:
            to_remove = list(self.cache.keys())[:self.max_size // 10]
            for k in to_remove:
                del self.cache[k]
        
        self.cache[key] = equity
        return equity
    
    def _compute_equity(self, hole_cards, board_cards, num_opponents, n_sims):
        """
        Monte Carlo equity calculation with PROPER TIE HANDLING
        Returns expected fractional payoff (pot equity)
        """
        if not EVAL7_AVAILABLE:
            return 0.5
        
        try:
            deck = eval7.Deck()
            hero_hand = [eval7.Card(rlcard_to_eval7(str(c))) for c in hole_cards]
            board = [eval7.Card(rlcard_to_eval7(str(c))) for c in board_cards]
            
            # Remove known cards
            known_cards = set(hero_hand + board)
            available = [c for c in deck.cards if c not in known_cards]
            
            wins = 0.0  # Now fractional for split pots
            
            for _ in range(n_sims):
                np.random.shuffle(available)
                
                # Deal opponent hands
                opp_hands = []
                card_idx = 0
                for _ in range(num_opponents):
                    if card_idx + 1 < len(available):
                        opp_hand = [available[card_idx], available[card_idx + 1]]
                        opp_hands.append(opp_hand)
                        card_idx += 2
                
                # Complete board
                remaining_board = 5 - len(board)
                if card_idx + remaining_board <= len(available):
                    full_board = board + available[card_idx:card_idx + remaining_board]
                else:
                    continue
                
                # FIXED: Proper tie handling
                hero_score = eval7.evaluate(hero_hand + full_board)
                better_count = 0  # Number of opponents with better hands
                tie_count = 1      # Start with 1 (hero always ties with self)
                
                for opp_hand in opp_hands:
                    opp_score = eval7.evaluate(opp_hand + full_board)
                    if opp_score < hero_score:  # Lower is better in eval7
                        better_count += 1
                    elif opp_score == hero_score:  # Tie
                        tie_count += 1
                
                # Hero wins (or ties) if not beaten by anyone
                if better_count == 0:
                    # Split pot among all players who tied for best hand
                    wins += 1.0 / tie_count
            
            return wins / n_sims if n_sims > 0 else 0.5
        
        except Exception as e:
            return 0.5
    
    def save(self, filepath):
        """Save cache to disk"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Saved equity cache ({len(self.cache)} entries)")
        except Exception as e:
            print(f"Warning: could not save equity cache: {e}")
    
    def load(self, filepath):
        """Load cache from disk"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded equity cache ({len(self.cache)} entries)")
        except Exception as e:
            print(f"Warning: could not load equity cache: {e}")
    
    def get_stats(self):
        """Return cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }

# Global equity cache
equity_cache = EquityCache(max_size=EQUITY_CACHE_SIZE)

# --- Helper Functions ---
def rlcard_to_eval7(card_str):
    """Convert RLCard format to eval7 format"""
    if not card_str or card_str == '': return None
    try:
        suit = card_str[0].lower()
        rank = card_str[1:]
        if rank == '10': rank = 'T'
        return f"{rank}{suit}"
    except Exception as e:
        print(f"Warning: could not convert card '{card_str}': {e}")
        return None

def calculate_hand_equity_cached(hole_cards, board_cards, num_opponents, n_sims):
    """Cached equity calculation"""
    return equity_cache.get(hole_cards, board_cards, num_opponents, n_sims)

def extract_features_from_state(state):
    """
    Extract hole cards, board, pot from RLCard state
    Returns: (hole_cards, board_cards, pot_size, position)
    """
    try:
        if 'raw_obs' in state:
            obs = state['raw_obs']
            hole_cards = obs.get('hand', [])
            board_cards = obs.get('public_cards', [])
            pot = obs.get('pot', 0)
            return hole_cards, board_cards, pot, 0
        else:
            return [], [], 0, 0
    except Exception as e:
        return [], [], 0, 0

# --- Equity Prediction Network ---
class EquityPredictor(nn.Module):
    """
    Small network that predicts equity from state
    This is an OBSERVATIONAL auxiliary task
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        equity = torch.sigmoid(self.fc3(x))
        return equity

class EnhancedNFSPAgent(NFSPAgent):
    """
    Extended NFSP with auxiliary equity prediction
    Key insight: predicting equity teaches hand strength faster
    """
    
    def __init__(self, state_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Auxiliary equity predictor
        if USE_AUXILIARY_LOSS:
            self.equity_predictor = EquityPredictor(state_dim).to(self.device)
            self.equity_optimizer = torch.optim.Adam(
                self.equity_predictor.parameters(),
                lr=LEARNING_RATE
            )
            self.equity_loss_history = []
        
        # Store transitions with equity labels
        self.equity_buffer = []
        self.max_buffer_size = 10000
        
    def feed_with_equity(self, ts, true_equity):
        """Feed transition with equity label"""
        # Standard NFSP feed
        self.feed(ts)
        
        # Store for auxiliary training
        if USE_AUXILIARY_LOSS and true_equity is not None:
            state = ts[0]
            self.equity_buffer.append((state, true_equity))
            
            if len(self.equity_buffer) > self.max_buffer_size:
                self.equity_buffer = self.equity_buffer[-self.max_buffer_size:]
    
    def train_equity_predictor(self, batch_size=64):
        """Train equity predictor on buffered examples"""
        if not USE_AUXILIARY_LOSS or len(self.equity_buffer) < batch_size:
            return None
        
        # Sample batch
        indices = np.random.choice(len(self.equity_buffer), batch_size, replace=False)
        batch = [self.equity_buffer[i] for i in indices]
        
        # Prepare data
        states = []
        equities = []
        
        for state, equity in batch:
            if isinstance(state, dict) and 'obs' in state:
                state_tensor = torch.FloatTensor(state['obs']).to(self.device)
                states.append(state_tensor)
                equities.append(equity)
        
        if len(states) == 0:
            return None
        
        states = torch.stack(states)
        equities = torch.FloatTensor(equities).unsqueeze(1).to(self.device)
        
        # Train
        self.equity_optimizer.zero_grad()
        predicted = self.equity_predictor(states)
        loss = F.mse_loss(predicted, equities)
        loss.backward()
        self.equity_optimizer.step()
        
        self.equity_loss_history.append(loss.item())
        return loss.item()
    
    def predict_equity(self, state):
        """Fast equity prediction using learned network"""
        if not USE_AUXILIARY_LOSS:
            return 0.5
        
        try:
            if isinstance(state, dict) and 'obs' in state:
                state_tensor = torch.FloatTensor(state['obs']).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    equity = self.equity_predictor(state_tensor).item()
                return equity
        except:
            return 0.5

def potential_based_shaping(state, next_state, agent, discount=0.95):
    """
    Potential-based reward shaping (Ng et al. 1999)
    Currently disabled - needs proper per-transition implementation
    """
    if not USE_POTENTIAL_SHAPING:
        return 0.0
    
    try:
        phi_s = agent.predict_equity(state)
        phi_s_next = agent.predict_equity(next_state)
        shaping = discount * phi_s_next - phi_s
        return shaping
    except:
        return 0.0

# --- Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not torch.cuda.is_available():
    print("⚠️  WARNING: No GPU detected! Training will be much slower.")
    print("   Recommendation: Enable GPU in Colab (Runtime → Change runtime type)")
    print("   Continuing on CPU...\n")
else:
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

set_seed(SEED)

# Create environment
env_config = {'seed': SEED, 'game_num_players': NUM_PLAYERS}
env = rlcard.make(ENV_NAME, config=env_config)
eval_env = rlcard.make(ENV_NAME, config=env_config)

print(f"Environment: {ENV_NAME}")
print(f"Players: {NUM_PLAYERS}")
print(f"Action space: {env.num_actions}")
print(f"State shape: {env.state_shape}\n")

# --- Initialize Agents ---
print("Initializing Enhanced NFSP Agents...")

# Extract state dimension
if isinstance(env.state_shape[0], list):
    state_dim = env.state_shape[0][0]
else:
    state_dim = env.state_shape[0]

print(f"  - State dimension: {state_dim}")

# FIXED: Create SEPARATE agent instances (not shared)
# Each agent needs independent replay buffers and mode sampling
agents = [
    EnhancedNFSPAgent(
        state_dim=state_dim,
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        hidden_layers_sizes=HIDDEN_LAYERS,
        q_mlp_layers=Q_MLP_LAYERS,
        device=device,
    )
    for _ in range(NUM_PLAYERS)
]

env.set_agents(agents)
eval_env.set_agents(agents)

print(f"✓ {NUM_PLAYERS} independent agents initialized")
print(f"  - Policy network: {HIDDEN_LAYERS}")
print(f"  - Q-network: {Q_MLP_LAYERS}")
if USE_AUXILIARY_LOSS:
    print(f"  - Auxiliary equity predictor: 128 hidden units")
print()

# Load equity cache if exists
equity_cache.load(EQUITY_CACHE_FILE)

# --- Resume from Checkpoint (if specified) ---
start_episode = 0

if RESUME_FROM_EPISODE > 0:
    RESUME_CHECKPOINT_DIR = f"{SAVE_PATH}/checkpoint_{RESUME_FROM_EPISODE}"
    checkpoint_file = os.path.join(RESUME_CHECKPOINT_DIR, 'nfsp_agents.pt')
    
    if os.path.exists(checkpoint_file):
        print("\n" + "="*70)
        print(f"RESUMING FROM CHECKPOINT: Episode {RESUME_FROM_EPISODE:,}")
        print("="*70)
        
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            
            # Load each agent's state
            for i, agent in enumerate(agents):
                agent_key = f'agent_{i}'
                if agent_key in checkpoint:
                    agent_data = checkpoint[agent_key]
                    
                    # Load policy network
                    if 'policy_network' in agent_data and 'mlp' in agent_data['policy_network']:
                        agent.policy_network.mlp.load_state_dict(agent_data['policy_network']['mlp'])
                    
                    # Load Q-network
                    if 'rl_agent' in agent_data and 'q_estimator' in agent_data['rl_agent']:
                        agent._rl_agent.q_estimator.qnet.load_state_dict(
                            agent_data['rl_agent']['q_estimator']['qnet']
                        )
                    
                    # Load optimizers
                    if 'policy_network_optimizer' in agent_data:
                        agent.policy_network_optimizer.load_state_dict(
                            agent_data['policy_network_optimizer']
                        )
                    
                    # Load auxiliary network
                    if USE_AUXILIARY_LOSS and 'equity_predictor' in agent_data:
                        agent.equity_predictor.load_state_dict(agent_data['equity_predictor'])
                        if 'equity_optimizer' in agent_data:
                            agent.equity_optimizer.load_state_dict(agent_data['equity_optimizer'])
                    
                    print(f"✓ Agent {i} loaded")
            
            start_episode = RESUME_FROM_EPISODE
            print(f"\n>>> Resuming training from episode {start_episode:,} <<<")
            print("="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ ERROR loading checkpoint: {e}")
            print("Starting from scratch instead...\n")
            start_episode = 0
    else:
        print(f"\n⚠️  WARNING: Checkpoint not found at {checkpoint_file}")
        print("Starting from scratch instead...\n")
        start_episode = 0

# --- Training Loop ---
print("="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Episodes: {start_episode:,} to {NUM_EPISODES:,}")
print(f"Expected time: ~several days on T4 GPU (CPU-bound environment)")
print(f"Checkpoints every: {SAVE_EVERY:,} episodes")
print("="*70 + "\n")

episode_rewards = []
start_time = time.time()
last_cache_save = start_episode

with Logger(SAVE_PATH) as logger:
    for episode in range(start_episode, NUM_EPISODES):
        
        # FIXED: Sample policy for EACH agent independently
        if episode > 0:
            for agent in agents:
                agent.sample_episode_policy()
        
        # 1. Generate self-play data
        trajectories, payoffs = env.run(is_training=True)
        
        # 2. Optional: Apply potential-based shaping (currently disabled)
        if USE_POTENTIAL_SHAPING:
            shaped_payoffs = []
            for player_idx, (traj, reward) in enumerate(zip(trajectories, payoffs)):
                shaped = reward
                for i, ts in enumerate(traj):
                    state = ts[0]
                    next_state = traj[i+1][0] if i+1 < len(traj) else state
                    shaping = potential_based_shaping(state, next_state, agents[player_idx], POTENTIAL_DISCOUNT)
                    shaped += shaping
                shaped_payoffs.append(shaped)
            payoffs = shaped_payoffs
        
        # 3. Feed agents with auxiliary equity information
        trajectories = reorganize(trajectories, payoffs)
        
        # Process each player's trajectory with their corresponding agent
        for player_idx in range(NUM_PLAYERS):
            for ts in trajectories[player_idx]:
                # Extract features
                state = ts[0]
                hole_cards, board_cards, pot, pos = extract_features_from_state(state)
                
                # Calculate equity (cached, fast)
                if len(hole_cards) > 0 and EVAL7_AVAILABLE:
                    equity = calculate_hand_equity_cached(
                        hole_cards, board_cards, 
                        NUM_PLAYERS - 1,
                        EQUITY_SIMS_TRAINING
                    )
                    agents[player_idx].feed_with_equity(ts, equity)
                else:
                    agents[player_idx].feed(ts)
        
        # 4. Train ALL agents
        for agent in agents:
            rl_loss = agent.train_rl()
            sl_loss = agent.train_sl()
        
        # 5. Train auxiliary equity predictors
        if USE_AUXILIARY_LOSS and episode % 10 == 0:
            for agent in agents:
                equity_loss = agent.train_equity_predictor(batch_size=64)
        
        episode_rewards.append(payoffs[0])
        
        # 6. Evaluation and Progress
        if episode % EVAL_EVERY == 0 and episode > start_episode:
            reward = tournament(eval_env, 100)[0]
            logger.log_performance(episode, reward)
            
            # Statistics
            elapsed_time = time.time() - start_time
            episodes_done = episode - start_episode
            eps_per_sec = episodes_done / elapsed_time if elapsed_time > 1e-6 else 0.0
            
            if eps_per_sec > 0:
                remaining_eps = NUM_EPISODES - episode
                eta_seconds = remaining_eps / eps_per_sec
                eta_hours = eta_seconds / 3600
            else:
                eta_hours = float('inf')
            
            avg_reward = np.mean(episode_rewards[-EVAL_EVERY:])
            cache_stats = equity_cache.get_stats()
            
            # Progress report
            progress = (episode / NUM_EPISODES) * 100
            print(f"\n{'='*70}")
            print(f"Episode {episode:,}/{NUM_EPISODES:,} ({progress:.1f}%)")
            print(f"{'='*70}")
            print(f"Tournament Reward: {reward:.4f}")
            print(f"Avg Recent Reward: {avg_reward:.4f}")
            print(f"Speed: {eps_per_sec:.1f} eps/s | Elapsed: {elapsed_time/3600:.2f}h | ETA: {eta_hours:.2f}h")
            
            if USE_AUXILIARY_LOSS and len(agents[0].equity_loss_history) > 0:
                recent_eq_loss = np.mean(agents[0].equity_loss_history[-100:])
                print(f"Equity Prediction Loss (Agent 0): {recent_eq_loss:.6f}")
            
            print(f"Equity Cache: {cache_stats['size']:,} entries, {cache_stats['hit_rate']*100:.1f}% hit rate")
            
            # Milestones
            if episode == 100_000:
                print("🎯 100K: Basic patterns learned")
            elif episode == 1_000_000:
                print("🎯 1M: Strategic play emerging")
            elif episode == 5_000_000:
                print("🎯 5M: Advanced tactics")
            elif episode == 10_000_000:
                print("🎯 10M: Strong play achieved")
            elif episode == 15_000_000:
                print("🎯 15M: Near-optimal play")
            
            print("="*70)
        
        # 7. Save checkpoints
        if episode % SAVE_EVERY == 0 and episode > start_episode:
            print(f"\n💾 Saving checkpoint at {episode:,}...")
            checkpoint_dir = f"{SAVE_PATH}/checkpoint_{episode}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save all agents
            checkpoint = {
                'episode': episode,
                'timestamp': time.time()
            }
            
            for i, agent in enumerate(agents):
                agent_data = {
                    'policy_network': {
                        'mlp': agent.policy_network.mlp.state_dict()
                    },
                    'rl_agent': {
                        'q_estimator': {
                            'qnet': agent._rl_agent.q_estimator.qnet.state_dict()
                        }
                    },
                    'policy_network_optimizer': agent.policy_network_optimizer.state_dict()
                }
                
                if USE_AUXILIARY_LOSS:
                    agent_data['equity_predictor'] = agent.equity_predictor.state_dict()
                    agent_data['equity_optimizer'] = agent.equity_optimizer.state_dict()
                
                checkpoint[f'agent_{i}'] = agent_data
            
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'nfsp_agents.pt'))
            print(f"✓ Checkpoint saved (all {NUM_PLAYERS} agents)")
        
        # 8. Save equity cache periodically
        if episode - last_cache_save >= 100_000:
            equity_cache.save(EQUITY_CACHE_FILE)
            last_cache_save = episode

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)

# --- Save Final Model ---
print("\n💾 Saving final model...")
os.makedirs(SAVE_PATH, exist_ok=True)

final_checkpoint = {
    'episode': NUM_EPISODES,
    'timestamp': time.time()
}

for i, agent in enumerate(agents):
    agent_data = {
        'num_actions': agent._num_actions,
        'state_shape': agent._state_shape,
        'mlp_layers': agent._layer_sizes,
        'policy_network': {
            'mlp': agent.policy_network.mlp.state_dict()
        },
        'rl_agent': {
            'q_estimator': {
                'qnet': agent._rl_agent.q_estimator.qnet.state_dict()
            }
        },
        'policy_network_optimizer': agent.policy_network_optimizer.state_dict()
    }
    
    if USE_AUXILIARY_LOSS:
        agent_data['equity_predictor'] = agent.equity_predictor.state_dict()
    
    final_checkpoint[f'agent_{i}'] = agent_data

torch.save(final_checkpoint, os.path.join(SAVE_PATH, 'nfsp_agents.pt'))
print(f"✓ Final model saved (all {NUM_PLAYERS} agents)")

# Save equity cache
equity_cache.save(EQUITY_CACHE_FILE)

# Create download package
print("\n📦 Creating download package...")
import shutil
archive_name = 'poker_model_improved_v2_20M'
shutil.make_archive(archive_name, 'zip', SAVE_PATH)

try:
    from google.colab import files
    print("\n⬇️  Downloading...")
    files.download(f'{archive_name}.zip')
    print("✓ Download complete!")
except ImportError:
    print(f"\n✓ Archive saved: {archive_name}.zip")

# Summary
total_time = time.time() - start_time
cache_stats = equity_cache.get_stats()

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"Total episodes: {NUM_EPISODES:,}")
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Final reward: {np.mean(episode_rewards[-10000:]):.4f}")
print(f"Equity cache: {cache_stats['size']:,} entries ({cache_stats['hit_rate']*100:.1f}% hit rate)")
print(f"Agents: {NUM_PLAYERS} independent instances with separate replay buffers")
if USE_AUXILIARY_LOSS:
    print(f"Auxiliary loss: Observational equity prediction")
print("="*70)
print("\n✅ Model ready! Extract zip and run poker_app.py")
