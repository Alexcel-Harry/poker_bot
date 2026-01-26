"""Interactive Poker Game v7 - Play against your trained bots with expanded actions!"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rlcard


# ============== V7 ACTION ABSTRACTION ==============
NUM_ABSTRACT_ACTIONS = 8
RAISE_SIZES = [0.0, 0.0, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0]

# Human-readable action names
ACTION_NAMES = [
    "FOLD ⚠️",
    "CHECK/CALL",
    "RAISE 0.33x pot (min)",
    "RAISE 0.5x pot",
    "RAISE 0.75x pot",
    "RAISE 1x pot",
    "RAISE 1.5x pot",
    "RAISE 2x pot / ALL-IN 🚨"
]


def get_abstract_legal_mask(legal_actions, num_actions=8):
    """Create mask for abstract action space based on what's legal in RLCard"""
    mask = np.zeros(num_actions, dtype=np.float32)
    legal_ids = list(legal_actions.keys())
    
    # Fold
    if 0 in legal_ids:
        mask[0] = 1.0
    
    # Check/Call
    if 1 in legal_ids:
        mask[1] = 1.0
    
    # Raises - if any raise is legal, all abstract raises are "legal"
    if any(a >= 2 for a in legal_ids):
        mask[2:] = 1.0
    
    return mask


def map_abstract_to_rlcard(abstract_action, legal_actions):
    """Map our expanded action to RLCard's available actions"""
    legal_ids = list(legal_actions.keys())
    
    # Direct mappings for fold/check/call
    if abstract_action == 0:  # fold
        return 0 if 0 in legal_ids else legal_ids[0]
    if abstract_action == 1:  # check/call
        return 1 if 1 in legal_ids else legal_ids[0]
    
    # Raise actions - find closest legal raise
    raise_actions = [a for a in legal_ids if a >= 2]
    if not raise_actions:
        return 1 if 1 in legal_ids else legal_ids[0]
    
    # Map based on relative size
    if abstract_action <= 3:  # Small raise (0.33-0.5x)
        return 2 if 2 in legal_ids else raise_actions[0]
    elif abstract_action <= 5:  # Medium raise (0.75-1x)
        return 3 if 3 in legal_ids else raise_actions[-1]
    else:  # Large raise (1.5-2x) / all-in
        return 4 if 4 in legal_ids else raise_actions[-1]


# ============== V7 NETWORK (4 layers) ==============
class StrategyNet(nn.Module):
    """V7 Strategy Network - supports both 3-layer (v6) and 4-layer (v7) architectures"""
    def __init__(self, state_dim, num_actions, hidden, num_layers=4):
        super().__init__()
        self.num_layers = num_layers
        
        self.fc1 = nn.Linear(state_dim, hidden[0])
        self.ln1 = nn.LayerNorm(hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.ln2 = nn.LayerNorm(hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.ln3 = nn.LayerNorm(hidden[2])
        
        if num_layers == 4:
            self.fc4 = nn.Linear(hidden[2], hidden[2] // 2)
            self.ln4 = nn.LayerNorm(hidden[2] // 2)
            self.out = nn.Linear(hidden[2] // 2, num_actions)
        else:
            self.out = nn.Linear(hidden[2], num_actions)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        if self.num_layers == 4:
            x = F.relu(self.ln4(self.fc4(x)))
        return F.softmax(self.out(x), dim=-1)


class PokerAgent:
    """Poker Agent supporting both v6 and v7 models"""
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.num_players = ckpt['num_players']
        self.state_dim = ckpt['state_dim']
        self.num_actions = ckpt['num_actions']
        hidden = ckpt['hidden_layers']
        
        # Detect model version based on action count and hidden layers
        self.is_v7 = self.num_actions == 8 or 'abstract_actions' in ckpt
        self.raise_sizes = ckpt.get('raise_sizes', RAISE_SIZES)
        
        # Determine number of layers based on hidden layer sizes
        num_layers = 4 if hidden[0] >= 512 else 3
        
        self.nets = [StrategyNet(self.state_dim, self.num_actions, hidden, num_layers).to(self.device)
                     for _ in range(self.num_players)]
        for i, sd in enumerate(ckpt['strat_nets']):
            self.nets[i].load_state_dict(sd)
            self.nets[i].eval()
        
        version = "v7 (expanded actions)" if self.is_v7 else "v6 (standard)"
        print(f"✅ Loaded {self.num_players}-player poker agent ({version})")
        print(f"   Network: {hidden}, Actions: {self.num_actions}")
    
    def get_action_probs(self, state, player_id):
        """Get probabilities for all actions"""
        obs = state.get('obs')
        if obs is None:
            return np.ones(self.num_actions) / self.num_actions
        
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            probs = self.nets[player_id](obs_t).cpu().numpy()[0]
        
        return probs
    
    def act(self, state, player_id, deterministic=False):
        """Select action for given state"""
        if state is None:
            return 0
        
        obs = state.get('obs')
        if obs is None:
            return 0
            
        legal = state.get('legal_actions', {})
        if not legal:
            return 0
        
        probs = self.get_action_probs(state, player_id)
        
        if self.is_v7:
            # V7: Use abstract action space
            mask = get_abstract_legal_mask(legal, self.num_actions)
            legal_abstract = np.where(mask > 0)[0]
            
            probs_legal = probs[legal_abstract]
            probs_legal = np.maximum(probs_legal, 0)
            prob_sum = probs_legal.sum()
            if prob_sum > 1e-8:
                probs_legal = probs_legal / prob_sum
            else:
                probs_legal = np.ones(len(legal_abstract)) / len(legal_abstract)
            
            if deterministic:
                abstract_action = legal_abstract[np.argmax(probs_legal)]
            else:
                abstract_action = np.random.choice(legal_abstract, p=probs_legal)
            
            return map_abstract_to_rlcard(abstract_action, legal), abstract_action
        else:
            # V6: Direct action space
            legal_ids = list(legal.keys())
            probs_legal = probs[legal_ids]
            probs_legal = probs_legal / probs_legal.sum()
            
            if deterministic:
                action = legal_ids[np.argmax(probs_legal)]
            else:
                action = np.random.choice(legal_ids, p=probs_legal)
            
            return action, action


def format_card(card_str):
    """Convert card string to emoji format"""
    if not card_str:
        return "??"
    
    card = str(card_str)
    
    # Suit emoji mapping
    suit_map = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣',
                's': '♠', 'h': '♥', 'd': '♦', 'c': '♣'}
    
    for suit_char, emoji in suit_map.items():
        if suit_char in card:
            card = card.replace(suit_char, emoji)
            break
    
    return card


def display_game_state(env, state, human_player=0):
    """Display the current game state"""
    print("\n" + "="*70)
    
    # Show pot
    pot = 0
    try:
        if hasattr(env.game, 'pot'):
            pot = sum(env.game.pot) if isinstance(env.game.pot, list) else env.game.pot
    except:
        pass
    print(f"💰 POT: ${pot}")
    
    # Show community cards
    public_cards = []
    try:
        if hasattr(env.game, 'public_cards') and env.game.public_cards:
            public_cards = env.game.public_cards
    except:
        pass
    
    stage_names = {0: "PRE-FLOP", 3: "FLOP", 4: "TURN", 5: "RIVER"}
    stage = stage_names.get(len(public_cards), "")
    
    if public_cards:
        cards = [format_card(str(card)) for card in public_cards]
        print(f"🃏 COMMUNITY CARDS ({stage}): {' '.join(cards)}")
    else:
        print(f"🃏 COMMUNITY CARDS (PRE-FLOP): [None yet]")
    
    print("-"*70)
    
    # Show all players
    for i in range(env.num_players):
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        
        try:
            chips = env.game.players[i].in_chips
            status = env.game.players[i].status
        except:
            chips = "?"
            status = "active"
        
        status_emoji = "✅" if status == "alive" else "❌" if status == "folded" else "💸" if status == "allin" else ""
        
        if i == human_player:
            hand_cards = []
            try:
                if hasattr(env.game.players[i], 'hand'):
                    hand_cards = env.game.players[i].hand
            except:
                pass
            
            print(f"{prefix} (Player {i}) - In pot: ${chips} - {status_emoji} {status}")
            if hand_cards:
                hand = [format_card(str(card)) for card in hand_cards]
                print(f"   🎴 Your cards: {' '.join(hand)}")
        else:
            print(f"{prefix} (Player {i}) - In pot: ${chips} - {status_emoji} {status}")
    
    print("="*70)


def get_human_action(state, agent):
    """Get action from human player with v7 abstract actions"""
    if state is None:
        return 0, 0
    
    legal_actions = state.get('legal_actions', {})
    if not legal_actions:
        return 0, 0
    
    print("\n🎯 YOUR TURN - Available actions:")
    print("-"*70)
    
    if agent.is_v7:
        # V7: Show abstract actions
        mask = get_abstract_legal_mask(legal_actions, agent.num_actions)
        legal_abstract = np.where(mask > 0)[0]
        
        # Get bot's probability assessment for reference
        probs = agent.get_action_probs(state, 0)
        
        action_list = []
        for idx, abstract_id in enumerate(legal_abstract):
            prob = probs[abstract_id] * 100
            action_list.append(abstract_id)
            
            # Show what RLCard action it maps to
            rlcard_action = map_abstract_to_rlcard(abstract_id, legal_actions)
            print(f"  [{idx}] {ACTION_NAMES[abstract_id]:<30} (Bot would: {prob:5.1f}%) → RLCard action {rlcard_action}")
        
        print("-"*70)
        print("💡 TIP: Bot probabilities show what the AI would do in this situation")
        print("-"*70)
        
        while True:
            try:
                choice = input(f"Enter your choice [0-{len(action_list)-1}]: ").strip()
                choice_idx = int(choice)
                if 0 <= choice_idx < len(action_list):
                    abstract_action = action_list[choice_idx]
                    rlcard_action = map_abstract_to_rlcard(abstract_action, legal_actions)
                    return rlcard_action, abstract_action
                else:
                    print(f"❌ Please enter a number between 0 and {len(action_list)-1}")
            except ValueError:
                print("❌ Invalid input. Please enter a number.")
            except KeyboardInterrupt:
                print("\n")
                return list(legal_actions.keys())[0], 0
    else:
        # V6: Direct actions
        action_list = list(legal_actions.keys())
        for idx, action_id in enumerate(action_list):
            print(f"  [{idx}] Action {action_id}")
        
        while True:
            try:
                choice = input(f"Enter your choice [0-{len(action_list)-1}]: ").strip()
                choice_idx = int(choice)
                if 0 <= choice_idx < len(action_list):
                    return action_list[choice_idx], action_list[choice_idx]
            except:
                pass


def play_game(model_path, human_player=0):
    """Main game loop"""
    print("\n" + "🎰"*35)
    print("    TEXAS HOLD'EM v7 - Expanded Action Space!")
    print("🎰"*35)
    
    # Load agent
    print("\n📦 Loading trained model...")
    agent = PokerAgent(model_path)
    
    # Create environment
    env = rlcard.make('no-limit-holdem', config={'game_num_players': agent.num_players})
    
    print(f"👤 You are Player {human_player}")
    print(f"🤖 Bots are Players {[i for i in range(agent.num_players) if i != human_player]}")
    
    if agent.is_v7:
        print("\n📋 V7 Action Space:")
        for i, name in enumerate(ACTION_NAMES):
            print(f"   {i}: {name}")
    
    input("\nPress ENTER to start the game...")
    
    # Start game
    state, current_player = env.reset()
    
    while not env.is_over():
        if state is None:
            try:
                state = env.get_state(current_player)
            except:
                break
        
        display_game_state(env, state, human_player)
        
        if current_player == human_player:
            rlcard_action, abstract_action = get_human_action(state, agent)
            if agent.is_v7:
                print(f"\n✅ You chose: {ACTION_NAMES[abstract_action]}")
            else:
                print(f"\n✅ You chose: Action {rlcard_action}")
        else:
            rlcard_action, abstract_action = agent.act(state, current_player)
            
            if agent.is_v7:
                probs = agent.get_action_probs(state, current_player)
                top_probs = sorted(enumerate(probs), key=lambda x: -x[1])[:3]
                
                print(f"\n🤖 BOT {current_player} thinking...")
                print(f"   Top choices: ", end="")
                for act_id, prob in top_probs:
                    if prob > 0.01:
                        print(f"{ACTION_NAMES[act_id].split()[0]}({prob*100:.0f}%) ", end="")
                print()
                print(f"   ➡️ Chose: {ACTION_NAMES[abstract_action]}")
            else:
                print(f"\n🤖 BOT {current_player} chose: Action {rlcard_action}")
            
            input("Press ENTER to continue...")
        
        try:
            state, current_player = env.step(rlcard_action)
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    
    # Game over
    print("\n" + "🏆"*35)
    print("              GAME OVER")
    print("🏆"*35)
    
    # Show final community cards
    try:
        if hasattr(env.game, 'public_cards') and env.game.public_cards:
            cards = [format_card(str(card)) for card in env.game.public_cards]
            print(f"\n🃏 FINAL BOARD: {' '.join(cards)}")
    except:
        pass
    
    # Show all hands
    print("\n" + "-"*70)
    print("SHOWDOWN - All Players' Cards:")
    for i in range(env.num_players):
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        hand_cards = []
        
        try:
            if hasattr(env.game.players[i], 'hand') and env.game.players[i].hand:
                hand_cards = env.game.players[i].hand
        except:
            pass
        
        status = "?"
        try:
            status = env.game.players[i].status
        except:
            pass
        
        if hand_cards:
            hand = [format_card(str(card)) for card in hand_cards]
            print(f"  {prefix}: {' '.join(hand)} ({status})")
        else:
            print(f"  {prefix}: [Hidden] ({status})")
    
    print("-"*70)
    
    # Results
    payoffs = env.get_payoffs()
    print("\n💰 FINAL RESULTS:")
    
    winner_idx = np.argmax(payoffs)
    for i, payoff in enumerate(payoffs):
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        result_emoji = "🏆" if i == winner_idx and payoff > 0 else "📉" if payoff < 0 else "➖"
        print(f"  {result_emoji} {prefix}: ${payoff:+.2f}")
    
    if payoffs[human_player] > 0:
        print("\n🎉🎉🎉 Congratulations! You won! 🎉🎉🎉")
    elif payoffs[human_player] < 0:
        print("\n😔 Better luck next time!")
    else:
        print("\n🤝 It's a tie!")
    
    print("="*70)


def main():
    import sys
    import os
    
    # Try multiple default paths
    default_paths = [
        'poker_deep_cfr_1.0/strategy_only.pt',
        'poker_deep_cfr_0.1/strategy_only.pt',
    ]
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = None
        for path in default_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if model_path is None or not os.path.exists(model_path):
        print("❌ Error: No model file found!")
        print("\nUsage: python play_poker_v7.py [model_path]")
        print(f"\nSearched paths:")
        for p in default_paths:
            print(f"  - {p}")
        return
    
    print(f"📂 Using model: {model_path}")
    
    while True:
        try:
            play_game(model_path, human_player=0)
            
            print("\n" + "="*70)
            again = input("Play again? (y/n): ").strip().lower()
            if again != 'y':
                print("\n👋 Thanks for playing! Goodbye!")
                break
        except KeyboardInterrupt:
            print("\n\n👋 Game interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()