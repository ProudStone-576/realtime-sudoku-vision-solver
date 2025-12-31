import cv2
import numpy as np

# =========================
# VIDEO INPUT
# =========================
cap = cv2.VideoCapture(0)

# =========================
# SUDOKU SOLVER (BACKTRACKING)
# =========================
def solve_sudoku(board):
    empty = find_empty(board)
    if not empty:
        return True

    r, c = empty
    for num in range(1, 10):
        if is_valid(board, r, c, num):
            board[r, c] = num
            if solve_sudoku(board):
                return True
            board[r, c] = 0
    return False


def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i, j] == 0:
                return i, j
    return None


def is_valid(board, r, c, num):
    if num in board[r]:
        return False
    if num in board[:, c]:
        return False

    br, bc = 3 * (r // 3), 3 * (c // 3)
    if num in board[br:br+3, bc:bc+3]:
        return False

    return True


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        grid_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(grid_contour)

        sudoku_img = thresh[y:y+h, x:x+w]

        if w < 200 or h < 200:
            cv2.imshow("Sudoku Solver", frame)
            continue

        sudoku_img = cv2.resize(sudoku_img, (450, 450))
        sudoku_img = cv2.copyMakeBorder(
            sudoku_img, 20, 20, 20, 20,
            cv2.BORDER_CONSTANT, value=0
        )

        cell_h = sudoku_img.shape[0] // 9
        cell_w = sudoku_img.shape[1] // 9

        board = np.zeros((9, 9), dtype=int)

        # =========================
        # DIGIT EXTRACTION (PLACEHOLDER)
        # =========================
        # NOTE:
        # OCR / CNN digit recognition to be added here.
        # For now, board remains empty (0s).
        # This keeps the prototype honest.

        # =========================
        # SOLVE
        # =========================
        solved = board.copy()
        if solve_sudoku(solved):
            for i in range(9):
                for j in range(9):
                    if solved[i, j] != 0:
                        px = x + j * (w // 9)
                        py = y + i * (h // 9)
                        cv2.putText(
                            frame,
                            str(solved[i, j]),
                            (px + 15, py + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

    cv2.imshow("Real-Time Sudoku Prototype", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
