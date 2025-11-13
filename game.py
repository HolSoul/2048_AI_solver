import random
import msvcrt
import os

class Game:
    def __init__(self, size=4):
        self.size = size
        self.board = [[0] * size for _ in range(size)]
        self.score = 0
        self.last_added_tile_info = None # Used by add_new_tile to pass info to move()
        
        # Initialize with two tiles, but clear animation events for these initial adds
        self.add_new_tile() 
        self.add_new_tile()
        # self.animation_events is cleared in move() before processing a new move.

    def _get_empty_cells(self):
        empty_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0:
                    empty_cells.append((r, c))
        return empty_cells

    def add_new_tile(self):
        """Adds a new tile (2 or 4) to an empty cell."""
        empty_cells = self._get_empty_cells()
        if not empty_cells:
            return False # Should not happen if game logic is correct (game over check)

        r, c = random.choice(empty_cells)
        new_value = 2 if random.random() < 0.9 else 4
        self.board[r][c] = new_value
        
        # Store info about the newly added tile for the 'appear' animation
        self.last_added_tile_info = {'type': 'appear', 'pos_rc': (r, c), 'value': new_value}
        return True
        
    def peek_move(self, direction: int):
        """
        Проверяет, изменит ли ход доску, не меняя ее состояния.
        Возвращает True, если доска изменится, иначе False.
        """
        import copy
        
        # Создаем глубокую копию текущего состояния игры
        temp_game = copy.deepcopy(self)
        
        # Пытаемся сделать ход на временной копии
        animation_events = temp_game.move(direction)
        
        # Если ход привел к каким-либо событиям (изменениям), он валиден
        return bool(animation_events)

    def _compress_row(self, row_data, row_index, current_move_events):
        """
        Compresses a single row (list of tile values) to the left.
        Appends 'move' events to current_move_events.
        Returns the new row and a boolean indicating if any change occurred.
        """
        new_row = [val for val in row_data if val != 0]
        new_row_padded = new_row + [0] * (self.size - len(new_row))
        changed = False

        # Determine moves by comparing new_row_padded to original row_data,
        # tracking tiles by their original positions to find their new ones.
        # This simplified version assumes tiles maintain order and just shift.
        # A more robust way would track individual tile IDs if they could cross.
        
        write_idx = 0
        for read_idx, tile_val in enumerate(row_data):
            if tile_val != 0:
                if read_idx != write_idx: # Tile moved
                    current_move_events.append({
                        'type': 'move', 
                        'from_rc': (row_index, read_idx), 
                        'to_rc': (row_index, write_idx),
                        'value': tile_val
                    })
                    changed = True
                write_idx += 1
        
        # If only order of zeros and numbers changed, but numbers stayed in same relative order
        # e.g. [0, 2, 0, 2] -> [2, 2, 0, 0]. The 'changed' from above might be True.
        # Check if the resulting numeric part of the row is different from original.
        if not changed and new_row_padded != row_data :
             changed = True # Handles cases like [0,2,0,0] -> [2,0,0,0]

        return new_row_padded, changed

    def _merge_row(self, row_data, row_index, current_move_events):
        """
        Merges tiles in a single row (assumed to be already compressed to the left for this logic).
        Appends 'merge' events and any subsequent 'move' events due to post-merge compression.
        Returns the final row state and a boolean indicating if any merge or related change occurred.
        """
        current_row = list(row_data) # Work on a copy
        merges_occurred = False
        
        i = 0
        while i < self.size - 1:
            if current_row[i] != 0 and current_row[i] == current_row[i+1]:
                val = current_row[i]
                merged_value = val * 2
                self.score += merged_value
                
                current_move_events.append({
                    'type': 'merge',
                    'from_rc1': (row_index, i),      # Tile that absorbs and changes value
                    'from_rc2': (row_index, i + 1),  # Tile that moves and disappears
                    'to_rc': (row_index, i),         # Final position of the merged tile
                    'original_value': val,           # Value of individual tiles before merge
                    'merged_value': merged_value
                })
                
                current_row[i] = merged_value
                current_row[i+1] = 0 # Tile at i+1 is now gone (merged into i)
                merges_occurred = True
                # Important: A tile can only participate in one merge per move.
                # So, after a merge at `i`, `i` should advance past the now-empty `i+1`.
                # However, the standard 2048 logic is that the merged tile cannot merge again in the same move.
                # e.g. [2,2,4,0] -> [4,0,4,0] (first 2s merge). Then [4,4,0,0] -> [8,0,0,0] is NOT in same move.
                # So, if current_row[i] merged, it cannot merge again with current_row[i+2] (now current_row[i+1] effectively after compression)
                # The current structure with iterating i and then a full re-compress handles this naturally.
                i += 1 # Advance i to check the next pair. If [2,2,2,2], first pair merges. Then i becomes 1 (value 0).
                       # Then i becomes 2, it will check current_row[2] and current_row[3].
            i += 1

        if merges_occurred:
            # After merges, the row might have new zeros (e.g., [4,4,2,0] -> merged [8,0,2,0]).
            # This row needs to be compressed again. The _compress_row will add new 'move' events.
            final_row, compress_changed_after_merge = self._compress_row(current_row, row_index, current_move_events)
            return final_row, True # If merges occurred, the board definitely changed.
            
        return current_row, False # No merges occurred

    def _process_move_left(self, current_animation_events):
        """
        Processes a 'left' move for the entire board.
        Populates current_animation_events with all move and merge events.
        Returns True if the board changed, False otherwise.
        """
        board_changed_overall = False
        new_board_state = []

        for r_idx in range(self.size):
            original_row = list(self.board[r_idx])
            
            # Pass 1: Compress existing tiles and record initial moves
            # Events are added to current_animation_events by _compress_row
            compressed_row, comp_changed = self._compress_row(original_row, r_idx, current_animation_events)
            if comp_changed:
                board_changed_overall = True
            
            # Pass 2: Merge tiles in the compressed_row.
            # _merge_row handles merges and any subsequent compression, adding all related events.
            final_row_for_board, merge_changed = self._merge_row(compressed_row, r_idx, current_animation_events)
            if merge_changed:
                board_changed_overall = True
            
            new_board_state.append(final_row_for_board)

        if board_changed_overall:
            self.board = new_board_state # Update the actual board
        
        return board_changed_overall

    def _transpose_board_and_events(self, events_list):
        self._transpose()
        for event in events_list:
            if 'pos_rc' in event: event['pos_rc'] = (event['pos_rc'][1], event['pos_rc'][0])
            if 'from_rc' in event: event['from_rc'] = (event['from_rc'][1], event['from_rc'][0])
            if 'to_rc' in event: event['to_rc'] = (event['to_rc'][1], event['to_rc'][0])
            if 'from_rc1' in event: event['from_rc1'] = (event['from_rc1'][1], event['from_rc1'][0])
            if 'from_rc2' in event: event['from_rc2'] = (event['from_rc2'][1], event['from_rc2'][0])
    
    def _reverse_rows_and_events(self, events_list):
        self._reverse_rows() # Reverses self.board rows
        for event in events_list:
            if 'pos_rc' in event: event['pos_rc'] = (event['pos_rc'][0], self.size - 1 - event['pos_rc'][1])
            if 'from_rc' in event: event['from_rc'] = (event['from_rc'][0], self.size - 1 - event['from_rc'][1])
            if 'to_rc' in event: event['to_rc'] = (event['to_rc'][0], self.size - 1 - event['to_rc'][1])
            if 'from_rc1' in event: event['from_rc1'] = (event['from_rc1'][0], self.size - 1 - event['from_rc1'][1])
            if 'from_rc2' in event: event['from_rc2'] = (event['from_rc2'][0], self.size - 1 - event['from_rc2'][1])

    def move(self, direction: int):
        """
        Processes a move in the given direction.
        direction: 0: Up, 1: Down, 2: Left, 3: Right
        Returns a list of animation events if the board changed, otherwise an empty list.
        """
        current_move_animation_events = []
        self.last_added_tile_info = None # Reset before move

        board_changed_by_transform = False

        if direction == 0: # Up
            self._transpose_board_and_events(current_move_animation_events) # Events list is empty, just transposes board
            board_changed_by_transform = self._process_move_left(current_move_animation_events)
            self._transpose_board_and_events(current_move_animation_events) # Transposes board back and corrects event coords
        elif direction == 1: # Down
            self._transpose_board_and_events(current_move_animation_events)
            self._reverse_rows_and_events(current_move_animation_events) # Reverses board rows
            board_changed_by_transform = self._process_move_left(current_move_animation_events)
            self._reverse_rows_and_events(current_move_animation_events) # Reverses back & corrects event coords
            self._transpose_board_and_events(current_move_animation_events) # Transposes back & corrects event coords
        elif direction == 2: # Left
            board_changed_by_transform = self._process_move_left(current_move_animation_events)
            # No coordinate remapping needed for events as _process_move_left works in final coordinates
        elif direction == 3: # Right
            self._reverse_rows_and_events(current_move_animation_events) # Reverses board rows
            board_changed_by_transform = self._process_move_left(current_move_animation_events)
            self._reverse_rows_and_events(current_move_animation_events) # Reverses back & corrects event coords
        
        if board_changed_by_transform:
            tile_added = self.add_new_tile() # This sets self.last_added_tile_info
            if tile_added and self.last_added_tile_info:
                current_move_animation_events.append(self.last_added_tile_info)
            return current_move_animation_events
        
        return [] # No change, no events

    def print_board(self):
        for row in self.board:
            print("\t".join(map(str, row)))
        print(f"Score: {self.score}")

    def _transpose(self):
        self.board = [list(row) for row in zip(*self.board)]

    def _reverse_rows(self):
        for i in range(self.size):
            self.board[i].reverse()

    def can_move(self):
        # Check for empty cells
        if self._get_empty_cells():
            return True
        # Check for possible merges
        for r in range(self.size):
            for c in range(self.size):
                val = self.board[r][c]
                if c + 1 < self.size and val == self.board[r][c+1]: return True
                if r + 1 < self.size and val == self.board[r+1][c]: return True
        return False

    def is_game_over(self):
        return not self.can_move()

# Main block for testing game.py (if needed, currently set up for msvcrt)
if __name__ == '__main__':
    game = Game()
    key_to_direction = { b'H': 0, b'P': 1, b'K': 2, b'M': 3 }

    def clear_screen(): os.system('cls' if os.name == 'nt' else 'clear')

    print("Welcome to 2048 (Console Test Mode)!")
    print("Use arrow keys to move. Press 'Q' to quit.")
    input("Press Enter to start...")

    while not game.is_game_over():
        clear_screen()
        game.print_board()
        print("\nAnimation Events for last move:")
        # This part is conceptual for console, real events are for Pygame
        # For now, we'll just print if events were generated.
        # In a real test, you might print the content of game.move()'s return.

        print("\nArrow keys: move, Q: quit")
        char = msvcrt.getch()
        if char == b'\xe0':  # Special key
            actual_key = msvcrt.getch()
            if actual_key in key_to_direction:
                direction = key_to_direction[actual_key]
                animation_events = game.move(direction) # game.move now returns events
                if not animation_events: # If list is empty, no change or no valid move
                    print("Move not possible or did not change the board.")
                    msvcrt.getch() 
                else:
                    print(f"Generated {len(animation_events)} animation events.") # Basic feedback
                    # for ev in animation_events: print(ev) # Detailed print if needed
            else:
                print("Invalid arrow key.")
                msvcrt.getch()
        elif char.lower() == b'q':
            print("Quitting.")
            break
    
    clear_screen()
    game.print_board()
    if game.is_game_over(): print("\nGame Over!")
    print(f"Final Score: {game.score}")
    input("Press Enter to exit...")