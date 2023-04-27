import pygame
import random

pygame.init()

# 화면 설정
screen_width = 800
screen_height = 700
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Tetris Game")

# 색상 설정
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# 블록 모양 설정
block_shapes = [
    [[1, 1, 1, 1]],
    [[2, 2], [2, 2]],
    [[3, 3, 0], [0, 3, 3]],
    [[0, 4, 4], [4, 4, 0]],
    [[5, 5, 5], [0, 5, 0]],
    [[6, 6, 6], [6, 0, 0]],
    [[7, 7, 7], [0, 0, 7]]
]

# 게임 관련 변수 설정
block_size = 30
board_width = 10
board_height = 20
board = [[0 for y in range(board_height)] for x in range(board_width)]
current_block = None
next_block = None
block_color = None
x_pos = board_width // 2
y_pos = 0
score = 0
font = pygame.font.SysFont("comicsansms", 30)

# 블록 생성 함수
def create_block():
    blocks = [
        [[1, 1, 1, 1]],                    # I
        [[1, 1, 0], [0, 1, 1]],            # Z
        [[0, 1, 1], [1, 1, 0]],            # S
        [[1, 1, 1], [0, 0, 1]],            # J
        [[1, 1, 1], [0, 1, 0]],            # T
        [[1, 1], [1, 1]],                  # O
        [[1, 1, 1], [1, 0, 0]]             # L
    ]
    return random.choice(blocks)

# 블록 이동 함수
def move_block(dx, dy):
    global x_pos, y_pos
    if check_collision(current_block, x_pos + dx, y_pos + dy):
        return False
    x_pos += dx
    y_pos += dy
    return True

# 충돌 검사 함수
def check_collision(shape, x, y):
    for i in range(len(shape)):
        for j in range(len(shape[i])):
            if shape[i][j] and (x + j < 0 or x + j >= board_width or y + i >= board_height or board[x + j][y + i]):
                return True
    return False

# 블록 회전 함수
def rotate_block():
    global current_block
    rotated_block = [[current_block[y][x] for y in range(len(current_block))] for x in range(len(current_block[0]) - 1, -1, -1)]
    if not check_collision(rotated_block, x_pos, y_pos):
        current_block = rotated_block

# 블록 고정 함수
def fix_block():
    global board, x_pos, y_pos, current_block, score
    for i in range(len(current_block)):
        for j in range(len(current_block[i])):
            if current_block[i][j]:
                board[x_pos + j][y_pos + i] = current_block[i][j] 
     # 한 줄이 가득 찼는지 검사
    for i in range(board_height):
        if all(board[j][i] for j in range(board_width)):
            for j in range(board_width):
                del board[j][i]
                board[j].insert(0, 0)
            score += 10

    x_pos = board_width // 2
    y_pos = 0
    current_block = create_block()

# 게임 루프
def game_loop():
    global x_pos, y_pos
    clock = pygame.time.Clock()
    fps = 30

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    move_block(-1, 0)
                elif event.key == pygame.K_RIGHT:
                    move_block(1, 0)
                elif event.key == pygame.K_DOWN:
                    move_block(0, 1)
                elif event.key == pygame.K_SPACE:
                    rotate_block()

        screen.fill(BLACK)

        # 블록 그리기
        for i in range(len(current_block)):
            for j in range(len(current_block[i])):
                if current_block[i][j]:
                    pygame.draw.rect(screen, block_color, (j * block_size + x_pos * block_size, i * block_size + y_pos * block_size, block_size, block_size))

        # 보드 그리기
        for i in range(board_width):
            for j in range(board_height):
                if board[i][j]:
                    pygame.draw.rect(screen, block_color, (i * block_size, j * block_size, block_size, block_size))

        # 점수 표시
        score_text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(score_text, (10, 10))

        # 블록 고정
        if not move_block(0, 1):
            fix_block()

        pygame.display.update()
        clock.tick(fps)

game_loop()           