import gym
from stable_baselines3 import DQN

# 環境の作成
env = gym.make('CartPole-v1')

# DQNモデルの初期化
model = DQN("MlpPolicy", env, verbose=1)

# モデルの訓練
model.learn(total_timesteps=10000)

# 訓練されたモデルを保存
model.save("dqn_cartpole")

# モデルのロード
# model = DQN.load("dqn_cartpole")

# テスト
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()

# 環境を閉じる
env.close()
