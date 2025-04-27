from pathlib import Path

class Config:
    # Board representation
    BOARD_SIZE = 8
    PIECE_TYPES = 6  # Pawn, Knight, Bishop, Rook, Queen, King
    PLAYER_CNT = 2   # White and Black
    
    # Input shape for neural network (8x8 board with 12 piece planes + 2 auxiliary planes)
    INPUT_SHAPE = (8, 8, 14)
    
    # Output is a 8x8x8x8 policy (for each starting square, each target square)
    # Simplified to 64x64 = 4096 possible moves
    OUTPUT_SIZE = 64 * 64
    
    # Model parameters
    NUM_FILTERS = 256
    NUM_RESIDUAL_BLOCKS = 10
    
    # Training parameters
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    EPOCHS_PER_ITERATION = 5
    VALIDATION_SPLIT = 0.2
    
    # Self-play parameters
    NUM_SELF_PLAY_GAMES = 1
    SIMULATIONS_PER_MOVE = 100  # Reduced for faster play
    MAX_MOVES = 1000  # Safety limit
    TEMPERATURE = 1.0  # Exploration parameter
    TEMPERATURE_THRESHOLD = 10  # After this many moves, use temperature = 0
    
    # MCTS parameters
    C_PUCT = 4.0  # Exploration constant
    DIRICHLET_NOISE_ALPHA = 0.3
    DIRICHLET_NOISE_EPSILON = 0.25
    
    # Stockfish parameters
    STOCKFISH_TIME_LIMIT = 0.1  # seconds
    
    # Directories
    BASE_DIR = Path("chess_rl")
    MODEL_DIR = BASE_DIR / "models"
    MEMORY_DIR = BASE_DIR / "memory"
    PLOTS_DIR = BASE_DIR / "plots"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        cls.MODEL_DIR.mkdir(exist_ok=True, parents=True)
        cls.MEMORY_DIR.mkdir(exist_ok=True, parents=True)
        cls.PLOTS_DIR.mkdir(exist_ok=True, parents=True)
