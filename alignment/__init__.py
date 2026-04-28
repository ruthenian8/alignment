"""Alignment package for producing aligned speech corpora."""

from alignment.align import AlignedSegment, align_srt_to_transcript
from alignment.export import ExportSegment
from alignment.index_parser import IndexRow, index_rows_to_dataframe, parse_index_plaintext
from alignment.srt import SrtSegment, format_srt, parse_srt
from alignment.transcript_parser import (
    TranscriptRecord,
    parse_transcript_file,
    transcript_records_to_dataframe,
)

__all__ = [
    "SrtSegment",
    "parse_srt",
    "format_srt",
    "IndexRow",
    "parse_index_plaintext",
    "index_rows_to_dataframe",
    "TranscriptRecord",
    "parse_transcript_file",
    "transcript_records_to_dataframe",
    "AlignedSegment",
    "align_srt_to_transcript",
    "ExportSegment",
]
