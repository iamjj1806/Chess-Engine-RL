import chess
import chess.engine
from config import Config

STOCKFISH_PATH = "C:\\Users\\shwet\\OneDrive\\Desktop\\stockfish\\stockfish-windows-x86-64-avx2.exe"

# ...existing code...
def get_stockfish_evaluation(board, time_limit=Config.STOCKFISH_TIME_LIMIT):
    """Get evaluation from Stockfish."""
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            result = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = result['score'].white().score(mate_score=10000)
            return score / 100  # Convert to range similar to our model's value
    except Exception as e:
        print(f"Stockfish error: {e}")
        return 0

def get_stockfish_move(board, time_limit=Config.STOCKFISH_TIME_LIMIT):
    """Get best move from Stockfish."""
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            return result.move
    except Exception as e:
        print(f"Stockfish error: {e}")
        return None
