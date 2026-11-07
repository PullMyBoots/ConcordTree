"""PHYLIP/PAML upload adapter; derived matrices stay in the job directory.

Each block is a locus. Accept relaxed sequential (including wrapped/PAML
name-only rows) and interleaved layouts. Do not guess gene boundaries from DNA.
"""
from __future__ import annotations

import re

DNA = frozenset('ACGTRYSWKMBDHVN?-.')
HEADER = re.compile(r'^(\d+)\s+(\d+)(?:\s+([IS]))?$', re.I)


def _sequence(text):
    value = ''.join(text.split()).upper()
    if not value or set(value) - DNA:
        raise ValueError('Use aligned DNA characters (A/C/G/T, IUPAC, N, ?, - or .).')
    return value


def _row(line):
    parts = line.split(maxsplit=1)
    name = parts[0]
    if len(name) > 256 or any(ord(c) < 33 for c in name):
        raise ValueError('Sequence names must be unique nonempty labels, at most 256 characters.')
    return name, _sequence(parts[1]) if len(parts) == 2 else ''


def _sequential(lines, start, n, length):
    rows = {}; pos = start
    for _ in range(n):
        if pos >= len(lines): raise ValueError('Fewer sequences than the PHYLIP header declares.')
        name, sequence = _row(lines[pos]); pos += 1
        if name in rows: raise ValueError('Duplicate sequence name within a locus.')
        while len(sequence) < length:
            if pos >= len(lines): raise ValueError('A sequence is shorter than its declared aligned length.')
            sequence += _sequence(lines[pos]); pos += 1
        if len(sequence) != length: raise ValueError('Sequence length differs from the PHYLIP header.')
        rows[name] = sequence
    return rows, pos


def _interleaved(lines, start, n, length):
    rows = {}; pos = start
    for _ in range(n):
        if pos >= len(lines): raise ValueError('Incomplete interleaved PHYLIP block.')
        name, sequence = _row(lines[pos]); pos += 1
        if name in rows or not sequence: raise ValueError('Interleaved PHYLIP needs one named row per taxon in its first block.')
        rows[name] = sequence
    while any(len(s) < length for s in rows.values()):
        for name in rows:
            if pos >= len(lines): raise ValueError('Incomplete interleaved PHYLIP block.')
            line = lines[pos]; pos += 1
            parts = line.split(maxsplit=1)
            # A continuation can be unnamed, or repeat exactly the same label.
            fragment = parts[1] if len(parts) == 2 and parts[0] == name else line
            rows[name] += _sequence(fragment)
            if len(rows[name]) > length: raise ValueError('Sequence exceeds its declared aligned length.')
    if any(len(s) != length for s in rows.values()):
        raise ValueError('Interleaved sequence lengths differ.')
    return rows, pos


def normalize_alignment(raw: bytes, task='gene', *, max_cells=100 * 1024 * 1024):
    """Return one normalized matrix, summary, and ordered locus lengths.

    A single species block uses msa_only; two or more use observed_loci.
    Different per-locus taxon sets are padded with unknown states, never zeros.
    """
    if task not in {'gene', 'species'}: raise ValueError('Choose Gene Tree or Species Tree.')
    try: text = raw.decode('utf-8-sig')
    except UnicodeDecodeError: raise ValueError('Use a UTF-8 PHYLIP/PAML text file.') from None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines: raise ValueError('The file is empty.')
    if lines[0].startswith('>'): raise ValueError('FASTA is not accepted. Upload PHYLIP for gene trees or PHYLIP/PAML locus blocks for species trees.')
    blocks = []; lengths = []; names = {}; observed = set(); pos = total = 0
    while pos < len(lines):
        header = HEADER.fullmatch(lines[pos])
        if not header: raise ValueError(f'Locus {len(blocks)+1}: expected a PHYLIP header: number of sequences, number of sites.')
        n, length = map(int, header.group(1, 2)); layout = (header.group(3) or '').upper(); pos += 1
        if n < 2 or length < 1: raise ValueError('Each locus needs at least 2 sequences and 1 aligned site.')
        if n * length > max_cells: raise ValueError('The alignment exceeds the matrix size limit.')
        readers = [_interleaved] if layout == 'I' else [_sequential] if layout == 'S' else [_sequential, _interleaved]
        failures = []
        for reader in readers:
            try:
                rows, end = reader(lines, pos, n, length)
                if end < len(lines) and not HEADER.fullmatch(lines[end]):
                    raise ValueError('Unexpected rows after the declared locus alignment.')
                break
            except ValueError as exc: failures.append(str(exc))
        else: raise ValueError(f'Locus {len(blocks)+1}: {failures[0]}')
        pos = end; total += length
        for name, sequence in rows.items():
            names.setdefault(name, None)
            if any(base in sequence for base in 'ACGT'): observed.add(name)
        if len(names) * total > max_cells: raise ValueError('The combined alignment exceeds the matrix size limit after filling missing taxa.')
        blocks.append(rows); lengths.append(length)
        if task == 'gene' and pos < len(lines):
            raise ValueError('Gene Tree accepts exactly one PHYLIP alignment. Use Species Tree for multiple locus blocks.')
    minimum = 4 if task == 'gene' else 24
    if len(names) < minimum: raise ValueError(f'{"Gene" if task == "gene" else "Species"} Tree requires at least {minimum} taxa across the input.')
    if set(names) - observed: raise ValueError('Every taxon needs at least one observed A/C/G/T base across its loci.')
    data = f'{len(names)} {total}\n' + ''.join(
        name + ' ' + ''.join((block[name] if name in block else '?' * length) for block, length in zip(blocks, lengths)) + '\n'
        for name in names)
    meta = dict(taxa=len(names), sites=total, loci=len(blocks), partitioned=task == 'species' and len(blocks) > 1)
    return data.encode(), meta, lengths
