import numpy as np
import chess
import chess.engine
from config import Config
from environment import ChessEnvironment
from mcts import MCTS
from model import create_model, save_model, load_model

STOCKFISH_PATH = "C:\\Users\\shwet\\OneDrive\\Desktop\\stockfish\\stockfish-windows-x86-64-avx2.exe"

def get_stockfish_evaluation(board, depth=20):
    """Get a Stockfish evaluation of the board position."""
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].relative.score(mate_score=10000)
        
        if score is None:
            return 0  # Neutral score if Stockfish fails to evaluate
        
        return normalize_stockfish_eval(score)

def normalize_stockfish_eval(eval_score):
    """Convert Stockfish centipawn values to a [-1, 1] range."""
    max_cp = 800  # Clip extreme values
    eval_score = max(-max_cp, min(max_cp, eval_score))  # Clip range
    return eval_score / max_cp  # Scale to [-1, 1]

def play_game(model, temperature_threshold=Config.TEMPERATURE_THRESHOLD):
    """Play a single game of self-play with Stockfish evaluations, displaying moves."""
    env = ChessEnvironment()
    mcts = MCTS(model, dirichlet_noise=True)
    states, policies, values = [], [], []
    
    move_count = 0
    while not env.is_game_over() and move_count < Config.MAX_MOVES:
        temperature = Config.TEMPERATURE if move_count < temperature_threshold else 0.0
        move, policy = mcts.search(env, Config.SIMULATIONS_PER_MOVE, temperature)
        
        # Convert move to chess.Move object if needed
        if isinstance(move, str) or isinstance(move, np.str_):
            move = chess.Move.from_uci(move)
        
        # Get Stockfish evaluation before the move
        stockfish_eval = get_stockfish_evaluation(env.board)
        
        # Print move details
        print(f"\nMove {move_count+1}: {env.board.san(move)} (Stockfish Eval: {stockfish_eval:.2f})")
        print(env.board)  # Display the board
        
        # Store state, policy, and Stockfish evaluation
        states.append(env.get_observation())
        policies.append(policy)
        values.append(stockfish_eval)
        
        env.step(move)
        move_count += 1
    
    print("\nGame Over!")
    return states, policies, values

def generate_self_play_data(model, num_games=Config.NUM_SELF_PLAY_GAMES):
    """Generate self-play games with Stockfish evaluations."""
    all_states, all_policies, all_values = [], [], []
    
    for i in range(num_games):
        print(f"\n===== Self-Play Game {i+1}/{num_games} =====")
        states, policies, values = play_game(model)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
    
    return all_states, all_policies, all_values

def save_self_play_data(states, policies, values, iteration):
    """Save self-play data to disk."""
    data = {'states': states, 'policies': policies, 'values': values}
    filename = Config.MEMORY_DIR / f"self_play_data_iter_{iteration}.npz"
    np.savez_compressed(filename, **data)
    print(f"Saved {len(states)} positions to {filename}")

def load_self_play_data(filename):
    """Load self-play data from disk."""
    data = np.load(filename, allow_pickle=True)
    return data['states'], data['policies'], data['values']

def load_all_self_play_data():
    """Load all self-play data from the memory directory."""
    all_states, all_policies, all_values = [], [], []
    
    for filename in Config.MEMORY_DIR.glob('*.npz'):
        try:
            states, policies, values = load_self_play_data(filename)
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return all_states, all_policies, all_values

def prepare_training_data(states, policies, values):
    """Prepare data for training the neural network."""
    X = np.array(states)  # Convert states to numpy array
    
    policy_indices, policy_values = [], []
    for i, policy in enumerate(policies):
        for move, prob in policy.items():
            try:
                from_square = chess.parse_square(move[:2])
                to_square = chess.parse_square(move[2:4])
                move_index = from_square * 64 + to_square
                policy_indices.append(move_index)
                policy_values.append(prob)
            except ValueError:
                print(f"Skipping invalid move: {move}")
                continue
    
    y_values = np.array(values).reshape(-1, 1)  # Convert values to numpy array
    
    return X, [np.array(policy_indices), np.array(policy_values)], y_values

def train_model(model, states, policies, values, validation_split=Config.VALIDATION_SPLIT):
    """Train the model on the provided data."""
    X, policy_data, y_values = prepare_training_data(states, policies, values)
    
    print(f"\nTraining on {len(X)} positions...")
    
    history = model.fit(
        X, [policy_data, y_values],
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS_PER_ITERATION,
        validation_split=validation_split,
        verbose=1
    )
    
    return history

def training_pipeline(model=None, iterations=10, games_per_iteration=Config.NUM_SELF_PLAY_GAMES):
    """Run the training pipeline with Stockfish evaluations."""
    if model is None:
        model_path = Config.MODEL_DIR / 'latest_model.h5'
        model = load_model(model_path) if model_path.exists() else create_model()
    
    for iteration in range(iterations):
        print(f"\n===== Iteration {iteration+1}/{iterations} =====")
        
        print(f"Generating self-play data ({games_per_iteration} games)...")
        states, policies, values = generate_self_play_data(model, games_per_iteration)
        save_self_play_data(states, policies, values, iteration)
        
        print("Loading all self-play data...")
        all_states, all_policies, all_values = load_all_self_play_data()
        print(f"Total positions: {len(all_states)}")
        
        print("Training model...")
        history = train_model(model, all_states, all_policies, all_values)
        
        iteration_model_path = Config.MODEL_DIR / f'model_iter_{iteration}.h5'
        latest_model_path = Config.MODEL_DIR / 'latest_model.h5'
        save_model(model, iteration_model_path)
        save_model(model, latest_model_path)
        print(f"Model saved to {iteration_model_path} and {latest_model_path}")
    
    print("Training pipeline completed!")
    return model