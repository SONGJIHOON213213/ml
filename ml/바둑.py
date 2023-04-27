import pygame
import sys

# 상수 정의
WINDOW_SIZE = (800, 800)  # 게임 창 크기
BOARD_SIZE = 600  # 바둑판 크기
STONE_SIZE = 30  # 바둑돌 크기
BLACK = (0, 0, 0)  # 검은색
WHITE = (255, 255, 255)  # 흰색

# 초기화
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Go Game")

# 바둑판 그리기
def draw_board():
    pygame.draw.rect(screen, BLACK, (100, 100, BOARD_SIZE, BOARD_SIZE), 2)
    for i in range(1, 20):
        pygame.draw.line(screen, BLACK, (100, 100 + i * BOARD_SIZE // 19), (100 + BOARD_SIZE, 100 + i * BOARD_SIZE // 19), 1)
        pygame.draw.line(screen, BLACK, (100 + i * BOARD_SIZE // 19, 100), (100 + i * BOARD_SIZE // 19, 100 + BOARD_SIZE), 1)

# 바둑돌 그리기
def draw_stone(color, stone):
    x, y = stone
    if color == 'black':
        pygame.draw.circle(screen, BLACK, (100 + x * BOARD_SIZE // 19, 100 + y * BOARD_SIZE // 19), STONE_SIZE // 2, 0)
    else:
        pygame.draw.circle(screen, WHITE, (100 + x * BOARD_SIZE // 19, 100 + y * BOARD_SIZE // 19), STONE_SIZE // 2, 0)

# 메인 루프
def main_loop():
    turn = 'black'
    stones = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                x = round((pos[0] - 100) * 19 / BOARD_SIZE)
                y = round((pos[1] - 100) * 19 / BOARD_SIZE)
                if (x, y) not in stones:
                    stones.append((x, y))
                    draw_stone(turn, (x, y))
                    if turn == 'black':
                        turn = 'white'
                    else:
                        turn = 'black'
        screen.fill(WHITE)
        draw_board()
        for stone in stones:
            if turn == 'black':
                draw_stone('black', stone)
            else:
                draw_stone('white', stone)
        pygame.display.update()