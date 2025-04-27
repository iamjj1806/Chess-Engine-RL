import tkinter as tk
from tkinter import ttk
import chess
import chess.svg
import threading
import time
import io
from PIL import Image, ImageTk
from wand.image import Image as WandImage  # Replacing cairosvg
import chess.engine

from config import Config
from environment import ChessEnvironment
from model import create_model, save_model, load_model
from mcts import MCTS
from training import training_pipeline


class ChessGUI:
    """Graphical interface for playing chess against the AI or Stockfish."""

    def __init__(self, master):
        self.master = master
        master.title("RL Chess")

        # Set up the frame first
        self.frame = ttk.Frame(master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Initialize status_var before using it
        self.status_var = tk.StringVar()
        self.status_var.set("Initializing...")
        self.status_label = ttk.Label(self.frame, textvariable=self.status_var)
        self.status_label.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Board canvas
        self.canvas = tk.Canvas(self.frame, width=400, height=400)
        self.canvas.grid(row=0, column=0, rowspan=6, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Set up the environment and model
        self.env = ChessEnvironment()
        self.model = None
        self.load_model()

        # Initialize MCTS
        self.mcts = MCTS(self.model, dirichlet_noise=False)

        # Initialize Stockfish with error handling
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(
                "C:\\Users\\jayes\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
            )
        except Exception as e:
            self.engine = None
            self.status_var.set(f"Error initializing Stockfish: {e}")

        # Control buttons
        self.new_game_button = ttk.Button(self.frame, text="New Game", command=self.new_game)
        self.new_game_button.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.train_button = ttk.Button(self.frame, text="Train Model", command=self.train_model_thread)
        self.train_button.grid(row=1, column=1, sticky=(tk.W, tk.E))

        self.play_white_button = ttk.Button(
            self.frame, text="Play as White", command=lambda: self.set_player_color(chess.WHITE)
        )
        self.play_white_button.grid(row=2, column=1, sticky=(tk.W, tk.E))

        self.play_black_button = ttk.Button(
            self.frame, text="Play as Black", command=lambda: self.set_player_color(chess.BLACK)
        )
        self.play_black_button.grid(row=3, column=1, sticky=(tk.W, tk.E))

        self.ai_vs_ai_button = ttk.Button(self.frame, text="AI vs AI", command=self.ai_vs_ai)
        self.ai_vs_ai_button.grid(row=4, column=1, sticky=(tk.W, tk.E))

        self.stockfish_button = ttk.Button(self.frame, text="Play vs Stockfish", command=self.play_vs_stockfish)
        self.stockfish_button.grid(row=5, column=1, sticky=(tk.W, tk.E))

        # Game state
        self.selected_square = None
        self.player_color = chess.WHITE
        self.ai_thinking = False

        # Bind events
        self.canvas.bind("<Button-1>", self.on_square_click)

        # Draw the initial board
        self.update_board()
        self.status_var.set("Ready to play")

    def load_model(self):
        """Load the latest trained model or create a new one if none exists."""
        try:
            model_path = Config.MODEL_DIR / 'latest_model.h5'
            if model_path.exists():
                self.model = load_model(model_path)
                self.status_var.set("Model loaded successfully")
            else:
                self.model = create_model()
                save_model(self.model, model_path)
                self.status_var.set("New model created")
        except Exception as e:
            self.status_var.set(f"Error loading model: {e}")
            self.model = create_model()

    def train_model_thread(self):
        """Start model training in a separate thread."""
        self.status_var.set("Training model...")
        self.train_button.state(['disabled'])

        def training():
            training_pipeline(self.model)
            self.status_var.set("Training complete")
            self.train_button.state(['!disabled'])

        threading.Thread(target=training).start()

    def set_player_color(self, color):
        """Set the player's color and start a new game."""
        self.player_color = color
        self.new_game()
        if color == chess.BLACK:
            self.make_ai_move()

    def make_ai_move(self):
        """Make a move using the AI."""
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.master.update()

        # Get move from MCTS
        move, _ = self.mcts.search(self.env, Config.SIMULATIONS_PER_MOVE)
        self.env.step(move)
        self.update_board()

        self.ai_thinking = False
        self.status_var.set("Your move")

    def on_square_click(self, event):
        """Handle clicks on chess squares."""
        if self.ai_thinking:
            return

        square_size = 400 // 8
        file_idx = event.x // square_size
        rank_idx = 7 - (event.y // square_size)
        clicked_square = chess.square(file_idx, rank_idx)

        if self.selected_square is None:
            piece = self.env.board.piece_at(clicked_square)
            if piece and piece.color == self.player_color:
                self.selected_square = clicked_square
                self.update_board(
                    highlight_moves=[
                        (clicked_square, move.to_square)
                        for move in self.env.board.legal_moves
                        if move.from_square == clicked_square
                    ]
                )
        else:
            move = chess.Move(self.selected_square, clicked_square)
            if move in self.env.board.legal_moves:
                self.env.step(move)
                self.selected_square = None
                self.update_board()

                if not self.env.is_game_over():
                    self.master.after(100, self.make_ai_move)  # Use after instead of threading
            else:
                self.selected_square = None
                self.update_board()

    def new_game(self):
        """Start a new game."""
        self.env = ChessEnvironment()
        self.selected_square = None
        self.update_board()
        self.status_var.set("New game started")

    def update_board(self, highlight_moves=None):
        """Update the chess board display."""
        svg_data = chess.svg.board(
            self.env.board,
            size=400,
            lastmove=self.env.last_move,
            check=self.env.board.king(self.env.board.turn) if self.env.board.is_check() else None,
            arrows=highlight_moves if highlight_moves else [],
        )

        # Convert SVG to PNG using Wand
        with WandImage(blob=svg_data.encode(), format="svg") as img:
            img.format = "png"
            png_data = img.make_blob()

        image = Image.open(io.BytesIO(png_data))
        photo = ImageTk.PhotoImage(image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.canvas.image = photo  # Keep a reference to prevent garbage collection

    def play_vs_stockfish(self):
        """Start a game against Stockfish."""
        if not self.engine:
            self.status_var.set("Stockfish not available")
            return

        self.new_game()
        self.player_color = chess.WHITE

        if self.player_color == chess.BLACK:
            self.stockfish_move()

    def stockfish_move(self):
        """Make a move using Stockfish."""
        if not self.engine:
            return

        self.ai_thinking = True
        self.status_var.set("Stockfish thinking...")
        self.master.update()

        result = self.engine.play(self.env.board, chess.engine.Limit(time=0.1))
        self.env.step(result.move)
        self.update_board()

        self.ai_thinking = False
        self.status_var.set("Your move")

    def ai_vs_ai(self):
        """Start a game between AI and Stockfish."""
        if not self.engine:
            self.status_var.set("Stockfish not available")
            return

        self.new_game()
        self.ai_thinking = True

        def play_game():
            while not self.env.is_game_over():
                if self.env.board.turn == chess.WHITE:
                    move, _ = self.mcts.search(self.env, Config.SIMULATIONS_PER_MOVE)
                else:
                    result = self.engine.play(self.env.board, chess.engine.Limit(time=0.1))
                    move = result.move

                self.env.step(move)
                self.update_board()
                time.sleep(0.5)

            self.ai_thinking = False
            self.status_var.set("Game over")

        threading.Thread(target=play_game).start()

    def __del__(self):
        """Cleanup Stockfish engine on exit."""
        if hasattr(self, 'engine') and self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass