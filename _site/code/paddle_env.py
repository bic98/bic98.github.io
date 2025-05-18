import numpy as np
import pygame


class PaddleEnv:
    def __init__(self, width=400, height=300):
        self.W, self.H = width, height
        self.BR = 10  # Ball radius
        self.PW, self.PH = 60, 10  # Paddle width/height
        self.FPS = 60
        self.paddle_y = self.H - self.PH
        self.action_space = 3
        self.state_dim = 5
        self.score_limit = 300
        self.reset()

    def reset(self):
        self.ball_pos = np.array([
            np.random.uniform(self.BR, self.W - self.BR),
            self.BR + 10
        ], dtype=np.float32)

        angle = np.random.uniform(-np.pi / 6, np.pi / 6)
        speed = 4.0
        self.ball_vel = np.array([
            speed * np.sin(angle),
            speed * np.cos(angle)
        ], dtype=np.float32)

        self.paddle_x = self.W / 2 - self.PW / 2
        self.done = False
        self.score = 0
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.ball_pos[0] / self.W,
            self.ball_pos[1] / self.H,
            self.ball_vel[0] / 10.0,
            self.ball_vel[1] / 10.0,
            self.paddle_x / self.W
        ], dtype=np.float32)

    def step(self, action: int):
        if action == 0:
            self.paddle_x -= 5
        elif action == 2:
            self.paddle_x += 5
        self.paddle_x = np.clip(self.paddle_x, 0, self.W - self.PW)

        self.ball_pos += self.ball_vel

        # 좌우 벽 충돌
        if self.ball_pos[0] <= 0:
            self.ball_pos[0] = self.BR
            self.ball_vel[0] *= -1
            if abs(self.ball_vel[1]) < 1.0:
                self.ball_vel[1] = np.sign(self.ball_vel[1]) * 2.0
        elif self.ball_pos[0] >= self.W - self.BR:
            self.ball_pos[0] = self.W - self.BR
            self.ball_vel[0] *= -1
            if abs(self.ball_vel[1]) < 1.0:
                self.ball_vel[1] = np.sign(self.ball_vel[1]) * 2.0

        # 천장 충돌
        if self.ball_pos[1] <= 0:
            self.ball_pos[1] = self.BR
            self.ball_vel[1] *= -1

        # 패들 충돌
        reward = 0
        ball_x, ball_y = self.ball_pos
        paddle_left = self.paddle_x
        paddle_right = self.paddle_x + self.PW
        paddle_top = self.paddle_y
        paddle_bottom = self.paddle_y + self.PH

        hit = (paddle_left - self.BR <= ball_x <= paddle_right + self.BR) and \
              (paddle_top - self.BR <= ball_y <= paddle_bottom)

        if hit:
            paddle_center = self.paddle_x + self.PW / 2
            offset = (ball_x - paddle_center) / (self.PW / 2)
            max_bounce_angle = np.radians(60)
            angle = offset * max_bounce_angle

            speed = max(3.5, np.linalg.norm(self.ball_vel))
            self.ball_vel[0] = speed * np.sin(angle)
            self.ball_vel[1] = -abs(speed * np.cos(angle))

            reward = 1
            self.score += 1

        # 바닥 충돌
        if self.ball_pos[1] >= self.H:
            reward = -1
            self.done = True

        # 점수 제한 도달 시 자동 종료
        if self.score >= self.score_limit:
            reward = 0
            self.done = True

        return self._get_state(), reward, self.done, {}

    def render(self, wait_after_done=1000):
        pygame.init()
        screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Paddle Game")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)

        running = True
        self.reset()
        while running:
            clock.tick(self.FPS)
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False

            if not self.done:
                action = 1
                _, _, self.done, _ = self.step(action)

            screen.fill((0, 0, 0))
            pygame.draw.circle(screen, (255, 255, 255),
                               self.ball_pos.astype(int), self.BR)
            pygame.draw.rect(screen, (255, 255, 255),
                             (int(self.paddle_x), self.paddle_y, self.PW, self.PH))
            screen.blit(font.render(
                f"Score: {self.score}", True, (255, 255, 255)), (10, 10))
            pygame.display.flip()

            if self.done:
                pygame.time.wait(wait_after_done)
                running = False

        pygame.quit()
