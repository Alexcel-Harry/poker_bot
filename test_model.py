"""Quick test script - watch bots play one hand"""

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
        
        print(f"✅ Loaded {self.num_players}-player poker agent")
    
    def act(self, state, player_id):
        obs = state['obs']
        legal = list(state.get('legal_actions', {}).keys())
        if not legal:
            return 0
        
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
            probs = self.nets[player_id](obs_t).cpu().numpy()[0]
        
        probs_legal = probs[legal]
        probs_legal = probs_legal / probs_legal.sum()
        return np.random.choice(legal, p=probs_legal)


def main():
    import sys
    import os
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'poker_deep_cfr/strategy_only.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Usage: python test_model.py [model_path]")
        return
    
    print("🎰 Testing your poker model...")
    print(f"📦 Loading model from: {model_path}\n")
    
    agent = PokerAgent(model_path)
    env = rlcard.make('no-limit-holdem', config={'game_num_players': agent.num_players})
    
    print(f"\n🎮 Running a test game with {agent.num_players} bots...\n")
    
    state, player = env.reset()
    step = 0
    
    while not env.is_over():
        action = agent.act(state, player)
        
        # Get action name
        action_names = ["CALL/CHECK", "RAISE(min)", "RAISE(2x)", "RAISE(3x)", 
                       "RAISE(4x)", "ALL-IN", "FOLD"]
        action_name = action_names[action] if action < len(action_names) else f"Action {action}"
        
        print(f"Step {step:2d} | Player {player} -> {action_name}")
        
        state, player = env.step(action)
        step += 1
    
    print(f"\n🏁 Game finished after {step} steps")
    print(f"💰 Final payoffs: {env.get_payoffs()}")
    print("\n✅ Model test successful! You're ready to play!")
    print("   Run: python play_poker.py")


if __name__ == '__main__':
    main()