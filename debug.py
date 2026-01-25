"""Debug script to see legal_actions structure"""
import rlcard
import json

env = rlcard.make('no-limit-holdem', config={'game_num_players': 3})
state, player = env.reset()

print("=" * 70)
print("LEGAL ACTIONS STRUCTURE DEBUG")
print("=" * 70)

print(f"\nCurrent player: {player}")

if 'legal_actions' in state:
    print(f"\nNumber of legal actions: {len(state['legal_actions'])}")
    print("\nLegal actions details:")
    print("-" * 70)
    
    for action_id, info in state['legal_actions'].items():
        print(f"\nAction ID: {action_id}")
        print(f"  Info type: {type(info)}")
        if isinstance(info, dict):
            print(f"  Info dict: {json.dumps(info, indent=4, default=str)}")
        else:
            print(f"  Info value: {info}")
else:
    print("\n⚠️ No 'legal_actions' in state")
    print(f"State keys: {state.keys()}")

print("\n" + "=" * 70)

# Let's try a few actions and see what happens
print("\nTesting action execution:")
print("-" * 70)

for i in range(3):
    state, player = env.reset()
    legal_actions = state.get('legal_actions', {})
    
    if legal_actions:
        first_action = list(legal_actions.keys())[0]
        print(f"\nRound {i+1}: Player {player}")
        print(f"  First legal action ID: {first_action}")
        print(f"  Action info: {legal_actions[first_action]}")
        
        # Execute the action
        state, next_player = env.step(first_action)
        print(f"  After action - Next player: {next_player}")
        print(f"  Player {player} status: {env.game.players[player].status}")