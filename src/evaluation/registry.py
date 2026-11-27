"""Portable scene layout; E04 is part of the books collection."""
from pathlib import Path
from src.intent.formal_evaluation import read_csv
def extra_books():return [],[]
def book_addition():return Path('evaluation/books')
def annotation_rows(dataset):return read_csv(Path(dataset)/'annotations.csv')
def result_dir(root,collection,clip):return Path(root)/collection/clip
