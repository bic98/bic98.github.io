import torch
import numpy as np
import pygame
from paddle_env import PaddleEnv
from train_r2d2 import R2D2Net  # LSTM 기반 QNet 사용

# 환경 및 모델 설정
env = PaddleEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R2D2Net(env.state_dim, env.action_space).to(device)
model.load_state_dict(torch.load("paddle_dqn_model.pth", map_location=device))
model.eval()

# 초기화
state = env.reset()
hidden = None  # LSTM hidden state 초기화
done = False

# pygame 시각화
pygame.init()
screen = pygame.display.set_mode((env.W, env.H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

while True:
    clock.tick(env.FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, D]
        with torch.no_grad():
            q_values, hidden = model(state_tensor, hidden)
            action = q_values[0, -1].argmax().item()
        state, _, done, _ = env.step(action)

    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 255, 255), env.ball_pos.astype(int), env.BR)
    pygame.draw.rect(screen, (255, 255, 255),
                     (int(env.paddle_x), env.paddle_y, env.PW, env.PH))
    screen.blit(font.render(f"Score: {env.score}", True, (255, 255, 255)), (10, 10))
    pygame.display.flip()

    if done:
        print(f"최종 점수: {env.score}")
        pygame.time.wait(2000)
        break
