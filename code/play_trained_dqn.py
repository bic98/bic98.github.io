# play_trained_dqn.py
import torch
import numpy as np
import pygame
from paddle_env import PaddleEnv
from train_dqn import QNet  # QNet 클래스 재사용

# 모델 로드
env = PaddleEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QNet(env.state_dim, env.action_space).to(device)
model.load_state_dict(torch.load("paddle_dqn_model.pth", map_location=device))
model.eval()

# 행동 선택 함수 (ε = 0 → 항상 argmax)


def select_action(state):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return model(state).argmax().item()


# 게임 루프
state = env.reset()
done = False
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
        action = select_action(state)
        state, _, done, _ = env.step(action)

    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 255, 255),
                       env.ball_pos.astype(int), env.BR)
    pygame.draw.rect(screen, (255, 255, 255),
                     (int(env.paddle_x), env.paddle_y, env.PW, env.PH))
    screen.blit(font.render(
        f"Score: {env.score}", True, (255, 255, 255)), (10, 10))
    pygame.display.flip()

    if done:
        print(f"🎮 최종 점수: {env.score}")
        pygame.time.wait(2000)
        break
