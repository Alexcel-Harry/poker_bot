"""Interactive Poker Game - Play against your trained bots!"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rlcard


class StrategyNet(nn.Module):
    def __init__(self, state_dim, num_actions, hidden):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden[0])
        self.ln1 = nn.LayerNorm(hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.ln2 = nn.LayerNorm(hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.ln3 = nn.LayerNorm(hidden[2])
        self.out = nn.Linear(hidden[2], num_actions)
    
    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return F.softmax(self.out(x), dim=-1)


class PokerAgent:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        ckpt = torch.load(model_path, map_location=self.device)
        self.num_players = ckpt['num_players']
        self.state_dim = ckpt['state_dim']
        self.num_actions = ckpt['num_actions']
        hidden = ckpt['hidden_layers']
        
        self.nets = [StrategyNet(self.state_dim, self.num_actions, hidden).to(self.device)
                     for _ in range(self.num_players)]
        for i, sd in enumerate(ckpt['strat_nets']):
            self.nets[i].load_state_dict(sd)
            self.nets[i].eval()
    
    def act(self, state, player_id):
        if state is None:
            return 0
        
        obs = state.get('obs')
        if obs is None:
            return 0
            
        legal = list(state.get('legal_actions', {}).keys())
        if not legal:
            return 0
        
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            probs = self.nets[player_id](obs_t).cpu().numpy()[0]
        
        probs_legal = probs[legal]
        probs_legal = probs_legal / probs_legal.sum()
        return np.random.choice(legal, p=probs_legal)


def format_card(card_str):
    """Convert card string to readable format"""
    if not card_str:
        return "??"
    return card_str


def display_game_state(env, state, human_player=0):
    """Display the current game state in a nice format"""
    print("\n" + "="*70)
    
    # Show pot - try multiple ways to access it
    pot = 0
    try:
        if hasattr(env.game, 'pot'):
            if isinstance(env.game.pot, list):
                pot = sum(env.game.pot)
            else:
                pot = env.game.pot
    except:
        pass
    print(f"💰 POT: ${pot}")
    
    # Show public cards if any
    public_cards = []
    try:
        if hasattr(env.game, 'public_cards') and env.game.public_cards:
            public_cards = env.game.public_cards
        elif 'public_cards' in state and state['public_cards']:
            public_cards = state['public_cards']
        elif 'raw_obs' in state and 'public_cards' in state['raw_obs']:
            public_cards = state['raw_obs']['public_cards']
    except:
        pass
    
    if public_cards:
        cards = [format_card(str(card)) for card in public_cards]
        print(f"🃏 COMMUNITY CARDS: {' '.join(cards)}")
    
    print("-"*70)
    
    # Show all players
    for i in range(env.num_players):
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        
        try:
            chips = env.game.players[i].in_chips
            status = env.game.players[i].status
        except:
            chips = "?"
            status = "?"
        
        # Show hand only for human player
        if i == human_player:
            hand_cards = []
            try:
                if hasattr(env.game.players[i], 'hand'):
                    hand_cards = env.game.players[i].hand
                elif 'raw_obs' in state and 'hand' in state['raw_obs']:
                    hand_cards = state['raw_obs']['hand']
            except:
                pass
            
            print(f"{prefix} (Player {i}) - Chips: ${chips} - Status: {status}")
            if hand_cards:
                hand = [format_card(str(card)) for card in hand_cards]
                print(f"   Your cards: {' '.join(hand)}")
        else:
            print(f"{prefix} (Player {i}) - Chips: ${chips} - Status: {status}")
    
    print("="*70)


def get_action_description(action_id, state):
    """Get human-readable action description from state"""
    if state is None:
        return f"Action {action_id}"
    
    # Try to get from raw_obs first
    if 'raw_obs' in state and 'legal_actions' in state['raw_obs']:
        raw_legal = state['raw_obs']['legal_actions']
        if isinstance(raw_legal, dict) and action_id in raw_legal:
            action_info = raw_legal[action_id]
            if isinstance(action_info, dict):
                action_type = action_info.get('action', '')
                amount = action_info.get('amount', 0)
                
                if action_type == 'fold':
                    return "FOLD"
                elif action_type == 'check':
                    return "CHECK"
                elif action_type == 'call':
                    return f"CALL ${amount}"
                elif action_type == 'raise':
                    return f"RAISE to ${amount}"
                elif action_type == 'all_in':
                    return f"ALL-IN ${amount}"
    
    # Fallback: CORRECTED action ID mapping based on observed behavior
    action_map = {
        0: "FOLD",              # CONFIRMED from gameplay
        1: "CALL/CHECK",        # Most likely
        2: "RAISE (small)",
        3: "RAISE (medium)",
        4: "ALL-IN",           # CONFIRMED from gameplay
        5: "RAISE (large)"
    }
    return action_map.get(action_id, f"Action {action_id}")


def get_action_name(action_id):
    """Legacy function - simple action ID to name mapping (less accurate)"""
    action_names = {
        0: "CALL/CHECK",
        1: "RAISE (min)",
        2: "RAISE (2x min)",
        3: "RAISE (3x min)", 
        4: "RAISE (4x min)",
        5: "RAISE (all-in)",
        6: "FOLD"
    }
    return action_names.get(action_id, f"Action {action_id}")


def get_human_action(state):
    """Get action from human player"""
    if state is None:
        print("⚠️ Warning: No state available, defaulting to action 0")
        return 0
    
    legal_actions = state.get('legal_actions', {})
    
    if not legal_actions:
        print("⚠️ Warning: No legal actions available, defaulting to action 0")
        return 0
    
    # Try to get action names from raw_obs if available
    action_names = {}
    if 'raw_obs' in state and 'legal_actions' in state['raw_obs']:
        raw_legal = state['raw_obs']['legal_actions']
        if isinstance(raw_legal, dict):
            action_names = raw_legal
    
    print("\n🎯 YOUR TURN - Available actions:")
    print("-"*70)
    
    action_list = []
    for idx, action_id in enumerate(legal_actions.keys()):
        # Try to get readable action name
        display = None
        
        # Method 1: Check raw_obs legal_actions
        if action_id in action_names:
            action_info = action_names[action_id]
            if isinstance(action_info, dict):
                action_type = action_info.get('action', '')
                amount = action_info.get('amount', 0)
                if action_type == 'fold':
                    display = "FOLD"
                elif action_type == 'check':
                    display = "CHECK"
                elif action_type == 'call':
                    display = f"CALL ${amount}"
                elif action_type == 'raise':
                    display = f"RAISE to ${amount}"
                elif action_type == 'all_in':
                    display = f"ALL-IN ${amount}"
        
        # Method 2: Use CORRECTED mapping based on actual game behavior
        if not display:
            action_map = {
                0: "FOLD ⚠️",           # CONFIRMED: Action 0 = FOLD!
                1: "CALL/CHECK",         # Most likely call or check
                2: "RAISE (small)",
                3: "RAISE (medium)",
                4: "ALL-IN 🚨",         # CONFIRMED: Action 4 = ALL-IN
                5: "RAISE (large)"
            }
            display = action_map.get(action_id, f"Action {action_id}")
        
        action_list.append(action_id)
        print(f"  [{idx}] {display} (ID: {action_id})")
    
    print("-"*70)
    print("⚠️  WARNING: Action 0 is FOLD! Be careful!")
    print("-"*70)
    
    while True:
        try:
            choice = input(f"Enter your choice [0-{len(action_list)-1}]: ").strip()
            choice_idx = int(choice)
            if 0 <= choice_idx < len(action_list):
                selected_action = action_list[choice_idx]
                return selected_action
            else:
                print(f"❌ Please enter a number between 0 and {len(action_list)-1}")
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input. Please enter a number.")
        except Exception as e:
            print(f"❌ Error: {e}")


def play_game(model_path, human_player=0):
    """Main game loop"""
    print("\n" + "🎰"*35)
    print("    TEXAS HOLD'EM - Play against your trained bots!")
    print("🎰"*35)
    
    # Load agent
    print("\n📦 Loading trained model...")
    agent = PokerAgent(model_path)
    
    # Create environment
    env = rlcard.make('no-limit-holdem', config={'game_num_players': agent.num_players})
    
    print(f"✅ Loaded {agent.num_players}-player poker agent")
    print(f"👤 You are Player {human_player}")
    print(f"🤖 Bots are Players {[i for i in range(agent.num_players) if i != human_player]}")
    
    input("\nPress ENTER to start the game...")
    
    # Start game
    state, current_player = env.reset()
    
    while not env.is_over():
        # Validate state
        if state is None:
            print("⚠️ Warning: Received None state, attempting to get state...")
            try:
                state = env.get_state(current_player)
            except:
                print("❌ Could not recover state, ending game")
                break
        
        # Display game state
        display_game_state(env, state, human_player)
        
        # Get action
        if current_player == human_player:
            # Human player's turn
            action = get_human_action(state)
            action_desc = get_action_description(action, state)
            print(f"\n✅ You chose: {action_desc}")
        else:
            # Bot's turn
            action = agent.act(state, current_player)
            action_desc = get_action_description(action, state)
            print(f"\n🤖 BOT {current_player} chose: {action_desc}")
            input("Press ENTER to continue...")
        
        # Execute action
        try:
            state, current_player = env.step(action)
        except Exception as e:
            print(f"❌ Error during step: {e}")
            break
    
    # Game over - show results
    print("\n" + "🏁"*35)
    print("              GAME OVER")
    print("🏁"*35)
    
    # Show final community cards
    public_cards = []
    try:
        if hasattr(env.game, 'public_cards') and env.game.public_cards:
            public_cards = env.game.public_cards
    except:
        pass
    
    if public_cards:
        cards = [format_card(str(card)) for card in public_cards]
        print(f"\n🃏 FINAL COMMUNITY CARDS: {' '.join(cards)}")
    
    # Show ALL players' hands (including folded players)
    print("\n" + "-"*70)
    print("ALL PLAYERS' CARDS:")
    for i in range(env.num_players):
        hand_cards = []
        
        # Try multiple ways to get the hand
        try:
            # Method 1: Direct access from player object
            if hasattr(env.game.players[i], 'hand') and env.game.players[i].hand:
                hand_cards = env.game.players[i].hand
            # Method 2: Try getting state for this player
            elif not hand_cards:
                try:
                    player_state = env.get_state(i)
                    if player_state and 'raw_obs' in player_state and 'hand' in player_state['raw_obs']:
                        hand_cards = player_state['raw_obs']['hand']
                except:
                    pass
        except:
            pass
        
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        status = "?" 
        try:
            status = env.game.players[i].status
        except:
            pass
            
        if hand_cards:
            hand = [format_card(str(card)) for card in hand_cards]
            print(f"{prefix} (Player {i}) - {status}: {' '.join(hand)}")
        else:
            print(f"{prefix} (Player {i}) - {status}: [Hand not visible]")
    print("-"*70)
    
    # Show payoffs
    payoffs = env.get_payoffs()
    print("\n💰 RESULTS:")
    for i, payoff in enumerate(payoffs):
        prefix = "👤 YOU" if i == human_player else f"🤖 BOT {i}"
        result = "WON" if payoff > 0 else "LOST" if payoff < 0 else "TIED"
        print(f"  {prefix} (Player {i}): ${payoff:+.2f} - {result}")
    
    if payoffs[human_player] > 0:
        print("\n🎉 Congratulations! You won!")
    elif payoffs[human_player] < 0:
        print("\n😔 Better luck next time!")
    else:
        print("\n🤝 It's a tie!")
    
    print("\n" + "="*70)


def main():
    import sys
    import os
    
    # Default model path
    default_path = 'poker_deep_cfr/strategy_only.pt'
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = default_path
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at '{model_path}'")
        print(f"\nUsage: python play_poker.py [model_path]")
        print(f"Default path: {default_path}")
        return
    
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
            print(f"\n❌ Error occurred: {e}")
            print("Exiting...")
            break


if __name__ == '__main__':
    main()