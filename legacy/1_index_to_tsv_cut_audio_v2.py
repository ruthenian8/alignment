#!/usr/bin/env python
# coding: utf-8
"""
Read a plaintext audio index.
Cut the audio by timestamps from the index.
Convert the plaintext index to tsv.
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Tag

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Optional: Google Colab specific import
try:
    from google.colab import files  # type: ignore
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Pre-compiled Regex Patterns
TIME_CODE_PATTERN = re.compile(r"[\d:\.,]{12}")
TIME_CODE_LINE_PATTERN = re.compile(r"^[\d:\.,]{12}[ -–]")
END_TIME_CODE_PATTERN = re.compile(r"[\d:\.,]{12}$")
CONTINUATION_KEYWORD_PATTERN = re.compile(r"продолж", re.IGNORECASE)


def read_docx_xml(path: Path) -> BeautifulSoup:
    """
    Read the XML document from a .docx file without extracting the full archive.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with zipfile.ZipFile(path, "r") as archive:
            with archive.open("word/document.xml") as doc_xml:
                content = doc_xml.read()
        return BeautifulSoup(content, "xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise OSError(f"Could not read 'word/document.xml' from {path}") from exc


def get_paragraph_texts(soup: BeautifulSoup) -> List[str]:
    """Extract raw paragraph texts from a Word XML soup."""
    paragraphs: List[Tag] = list(soup.find_all("w:p"))
    return [p.get_text() for p in paragraphs]


def parse_transcript_codes(paragraphs: List[str]) -> List[Dict[str, Any]]:
    """
    Extract timing codes, continuation flags, and transcription status.
    """
    # Filter for lines starting with a timestamp
    lines_with_codes = [
        line for line in paragraphs 
        if TIME_CODE_LINE_PATTERN.match(line.strip())
    ]

    records: List[Dict[str, Any]] = []

    for line in lines_with_codes:
        stripped = line.strip()
        
        # Extract Start Code
        start_match = TIME_CODE_PATTERN.match(stripped)
        if not start_match:
            continue
        start_code = start_match.group().replace(",", ".")

        # Check Transcription Status
        is_transcribed = "РАСПИСАНО" not in line

        # Check for Continuation
        # Logic: Clean trailing chars -> find timestamp at end -> check for keyword
        cont_code = ""
        cleaned_line = re.sub(r"[^\d\w]*$", "", line)
        end_match = END_TIME_CODE_PATTERN.search(cleaned_line)
        
        if end_match and CONTINUATION_KEYWORD_PATTERN.search(line):
            cont_code = end_match.group().replace(",", ".")

        records.append({
            "start": start_code,
            "trans": is_transcribed,
            "cont": cont_code,
            "prev": "",  # To be filled by DataFrame transformation
            "text": line,
        })

    return records


def link_continuations(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a DataFrame and link segments where 'cont' matches a previous 'start'.
    """
    df = pd.DataFrame.from_records(records)
    
    for idx, row in df.iterrows():
        cont_val = row.get("cont")
        if cont_val:
            # Find the index of the row that started this continuation
            matches = df.index[df["start"] == cont_val]
            if not matches.empty:
                df.loc[matches, "prev"] = idx
                
    return df


class AudioMapper:
    """
    Handles the splitting and re-concatenation of audio files based on DOCX transcripts.
    """
    def __init__(self, audio_file: Path | str) -> None:
        self.audio_path = Path(audio_file).resolve()
        if not self.audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

        self.base_name = self.audio_path.stem
        self.suffix = self.audio_path.suffix
        
        # Directory to store segments (e.g., ./AudioFile/...)
        self.work_dir = self.audio_path.parent / self.base_name
        
        self.table: Optional[pd.DataFrame] = None
        self.is_processed: bool = False

        # Initialize by parsing the transcript
        self._parse_transcript()

    def _ensure_table(self) -> pd.DataFrame:
        if self.table is None:
            raise RuntimeError("Transcript table not initialized.")
        return self.table

    def _parse_transcript(self, save_excel: bool = False) -> None:
        """Parses the corresponding .docx file."""
        docx_path = self.audio_path.with_suffix(".docx")
        
        try:
            soup = read_docx_xml(docx_path)
            raw_texts = get_paragraph_texts(soup)
            records = parse_transcript_codes(raw_texts)
        except Exception as e:
            raise RuntimeError(f"Failed to parse transcript {docx_path}: {e}")

        # Generate output filenames for each segment
        for idx, record in enumerate(records):
            record["name"] = f"{self.base_name}No{idx}{self.suffix}"

        self.table = link_continuations(records)
        
        if save_excel:
            self.save_excel()

    def split_audio(self, download: bool = False) -> None:
        """Splits the audio file using FFMPEG based on parsed timestamps."""
        df = self._ensure_table()
        names = df["name"].tolist()
        starts = df["start"].tolist()

        if not names:
            logging.warning(f"No valid timestamps found for {self.base_name}")
            return

        self.work_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Splitting {self.base_name} into {len(names)} segments...")

        # Process all segments
        for i, output_name in enumerate(names):
            start_time = starts[i]
            output_path = self.work_dir / output_name
            
            # Determine end time (None if it's the last segment)
            end_time = starts[i+1] if i < len(names) - 1 else None
            
            self._run_ffmpeg_split(start_time, end_time, output_path)

        self.is_processed = True

        if download and IN_COLAB:
            self._colab_download()

    def _run_ffmpeg_split(self, start: str, end: Optional[str], output: Path) -> None:
        cmd = ["ffmpeg", "-y", "-ss", start, "-i", str(self.audio_path)]
        if end:
            cmd.extend(["-to", end])
        
        cmd.extend(["-c", "copy", "-avoid_negative_ts", "1", str(output)])
        
        subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL, 
            check=True
        )

    def reverse_concat(self, save_excel: bool = False) -> None:
        """
        Concatenates segments backwards based on the 'prev' pointer.
        (Merging continuation files into their original parents).
        """
        if not self.is_processed:
            return

        df = self._ensure_table()
        # Sort descending to process end-of-chain items first
        reverse_df = df.sort_index(ascending=False)
        
        temp_list_file = self.work_dir / "concat_list.txt"

        for _, row in reverse_df.iterrows():
            prev_idx = row.get("prev")
            
            # Skip if no previous link or if prev is empty string/NaN
            if prev_idx == "" or pd.isna(prev_idx):
                continue
            
            try:
                # Resolve filenames
                prev_name = df.loc[prev_idx, "name"] # type: ignore
                curr_name = row["name"]
            except KeyError:
                continue

            prev_path = self.work_dir / prev_name
            curr_path = self.work_dir / curr_name
            
            if not prev_path.exists() or not curr_path.exists():
                logging.warning(f"Missing files for concat: {prev_name} + {curr_name}")
                continue

            # Create FFMPEG concat list
            # We use .resolve() to provide absolute paths to ffmpeg
            with open(temp_list_file, "w", encoding="utf-8") as f:
                f.write(f"file '{prev_path.resolve()}'\n")
                f.write(f"file '{curr_path.resolve()}'\n")

            temp_output = self.work_dir / f"temp_{prev_name}"
            
            # Run concatenation
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(temp_list_file), "-c", "copy", str(temp_output)
            ]
            
            logging.info(f"Concatenating: {curr_name} -> {prev_name}")
            try:
                subprocess.run(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL, 
                    check=True
                )
                # Overwrite the original parent file with the concatenated result
                temp_output.replace(prev_path)
            except subprocess.CalledProcessError:
                logging.error(f"FFMPEG failed to concatenate {curr_name}")

        # Clean up the text file
        if temp_list_file.exists():
            temp_list_file.unlink()

        logging.info("Concatenation complete.")
        self._cleanup_orphans()
        
        if save_excel:
            self.save_excel()

    def _cleanup_orphans(self) -> None:
        """Deletes segments that were merged into others."""
        df = self._ensure_table()
        # The 'keepers' are those that do not point to a previous file (roots)
        self.table = df.loc[df["prev"] == ""]
        
        valid_files = set(self.table["name"].values)
        
        if self.work_dir.exists():
            for file_path in self.work_dir.iterdir():
                if file_path.name not in valid_files and file_path.suffix == self.suffix:
                    file_path.unlink()

    def save_excel(self) -> None:
        """Saves the current state of the table to Excel."""
        if self.table is not None:
            output_path = self.work_dir / f"{self.base_name}.xlsx"
            self.table.to_excel(output_path, index=True)

    def _colab_download(self) -> None:
        """Zips and downloads the result (Colab only)."""
        archive_path = self.work_dir.with_suffix(".zip")
        subprocess.run(
            ["zip", "-r", str(archive_path.name), self.work_dir.name],
            cwd=str(self.work_dir.parent),
            check=True
        )
        if files:
            files.download(str(archive_path))


def main(directory: str) -> None:
    base_dir = Path(directory)
    if not base_dir.is_dir():
        logging.error(f"Directory not found: {base_dir}")
        sys.exit(1)

    files_found = 0
    for audio_path in base_dir.iterdir():
        if audio_path.is_file() and audio_path.suffix.lower() in {".wma", ".mp3"}:
            files_found += 1
            logging.info(f"Processing: {audio_path.name}")
            try:
                mapper = AudioMapper(audio_path)
                mapper.split_audio()
                mapper.reverse_concat(save_excel=True)
            except Exception as e:
                logging.error(f"Failed to process {audio_path.name}: {e}")
                # We continue to the next file rather than exiting
    
    if files_found == 0:
        logging.warning("No .wma or .mp3 files found in directory.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory_path>")
        sys.exit(1)
    
    main(sys.argv[1])
