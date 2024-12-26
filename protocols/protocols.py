from collections.abc import Iterable
from typing import Any
from typing import Protocol as ProtocolistProtocol
from typing import runtime_checkable
from typing import Union

CharSequence = Union[str, bytes, bytearray]
@runtime_checkable
class Board(ProtocolistProtocol):
	legal_moves: Iterable
	turn: Any
	def can_claim_fifty_moves(self):
		...
	def can_claim_threefold_repetition(self):
		...
	def copy(self, stack: bool):
		...
	def fen(self):
		...
	def has_kingside_castling_rights(self, arg0):
		...
	def has_queenside_castling_rights(self, arg0):
		...
	def is_checkmate(self):
		...
	def is_insufficient_material(self):
		...
	def is_stalemate(self):
		...
	def piece_at(self, arg0):
		...
	def piece_map(self):
		...
	def push(self, arg0: Any):
		...
@runtime_checkable
class Font(ProtocolistProtocol):
	
	def render(self, arg0: str, arg1: bool, arg2: Union[Any, tuple[int, int, int]]):
		...
@runtime_checkable
class HistorySubscript(ProtocolistProtocol):
	
	def fen(self):
		...
@runtime_checkable
class Model(ProtocolistProtocol):
	
	def predict(self, arg0: Any):
		...
@runtime_checkable
class Screen(ProtocolistProtocol):
	
	def blit(self, arg0: Union[Any, str], arg1: Union[Union[Any, tuple[Any, Any]], Union[tuple[int, Any], tuple[int, int]], tuple[int, Any], tuple[int, int]]):
		...
@runtime_checkable
class Win(ProtocolistProtocol):
	
	def blit(self, arg0, arg1: tuple[int, int]):
		...
