"""Quick RL training run to collect real metrics for experiments.md."""
import sys, os, time
# 把项目根目录加入 path（脚本在 scripts/ 下）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch, numpy as np, random
from rl_agent import PetEnvironment, DQN, ReplayBuffer

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

env = PetEnvironment(seed=seed)
agent = DQN(state_size=env.state_size, action_size=env.action_size, seed=seed)
target_net = DQN(state_size=env.state_size, action_size=env.action_size, seed=seed)
target_net.load_state_dict(agent.state_dict())
target_net.eval()
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
buf = ReplayBuffer(10000)

episodes = 40
gamma = 0.95
eps = 1.0
eps_min = 0.05
eps_decay = 0.995
target_update = 10
max_steps = 20  # 限制每 episode 步数，加速训练

rewards = []
losses = []

t0 = time.time()
for ep in range(episodes):
    s = env.reset()
    ep_r = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        a = agent.act(s, eps)
        ns, r, done, _ = env.step(a)
        buf.push(s, a, r, ns, done)
        s = ns
        ep_r += r
        steps += 1
        if len(buf) >= 64:
            bs, ba, br, bns, bd = buf.sample(64)
            st = torch.from_numpy(bs).float()
            at = torch.from_numpy(ba).long().unsqueeze(1)
            rt = torch.from_numpy(br).float().unsqueeze(1)
            nst = torch.from_numpy(bns).float()
            dt = torch.from_numpy(bd).float().unsqueeze(1)
            q = agent.forward(st).gather(1, at)
            q_next = target_net(nst).max(1)[0].unsqueeze(1)
            tq = rt + gamma * q_next * (1 - dt)
            loss = torch.nn.functional.mse_loss(q, tq)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    if ep % target_update == 0:
        agent.soft_update(target_net, tau=1.0)
    eps = max(eps_min, eps * eps_decay)
    rewards.append(ep_r)
    if (ep + 1) % 10 == 0:
        avg10 = np.mean(rewards[-10:])
        avg_loss = np.mean(losses[-50:]) if losses else 0.0
        print(f"Episode {ep+1:3d}/{episodes} | reward={ep_r:7.3f} | avg10={avg10:7.3f} | eps={eps:.3f} | loss={avg_loss:.4f}", flush=True)

elapsed = time.time() - t0
print(f"\n=== DONE ===")
print(f"episodes: {episodes}")
print(f"elapsed: {elapsed:.1f}s")
print(f"final_avg_reward (last 10): {np.mean(rewards[-10:]):.3f}")
print(f"all_time_avg: {np.mean(rewards):.3f}")
print(f"min reward: {min(rewards):.3f}")
print(f"max reward: {max(rewards):.3f}")
print(f"first 10 avg: {np.mean(rewards[:10]):.3f}")
print(f"last 10 avg: {np.mean(rewards[-10:]):.3f}")
print(f"improvement: {np.mean(rewards[-10:]) - np.mean(rewards[:10]):.3f}")
print(f"final epsilon: {eps:.3f}")
print(f"final avg loss: {np.mean(losses[-50:]):.4f}")
print(f"buffer size: {len(buf)}")

# Save checkpoint
os.makedirs("checkpoints", exist_ok=True)
torch.save({
    "policy_state_dict": agent.state_dict(),
    "target_state_dict": target_net.state_dict(),
    "episode": episodes,
    "final_epsilon": eps,
    "final_avg_reward": float(np.mean(rewards[-10:])),
}, "checkpoints/dqn_best.pt")
print("checkpoint saved -> checkpoints/dqn_best.pt")
