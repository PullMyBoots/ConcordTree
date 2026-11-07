"""Read original input, writing any format conversion only into the job output."""
from pathlib import Path
import re
import subprocess

DNA = re.compile(r'^[ACGTRYSWKMBDHVN? .-]*$', re.I)

def validate_phylip(path):
    # The frozen native reader expects one complete named taxon per line and
    # otherwise silently pads/truncates malformed rows. Validate before parsing.
    with Path(path).open() as handle:
        header = handle.readline().split()
        if len(header) != 2:
            raise ValueError('PHYLIP requires a taxa/sites header on the first line')
        try:
            count, length = map(int, header)
        except ValueError:
            raise ValueError('PHYLIP taxa/sites must be integers') from None
        if count < 4 or length < 1:
            raise ValueError('at least four taxa and one site required')
        names = set()
        for line in handle:
            if not line.strip():continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError('sequential PHYLIP requires one name and complete sequence per line')
            name, sequence = parts
            sequence = ''.join(sequence.split())
            if name in names or len(sequence) != length or not DNA.fullmatch(sequence):
                raise ValueError('invalid PHYLIP names, DNA characters, or sequence length')
            names.add(name)
        if len(names) != count:
            raise ValueError('PHYLIP row count differs from header')

def _fasta_rows(path):
    name = None
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:continue
            if line.startswith('>'):
                name = line[1:].strip()
                if not name or any(c.isspace() for c in name):
                    raise ValueError('FASTA records need nonempty names without whitespace')
                yield name, None
            else:
                if name is None:raise ValueError('sequence before first FASTA header')
                sequence = ''.join(line.split()).upper()
                if not DNA.fullmatch(sequence):raise ValueError('DNA alignment required')
                yield None, sequence

def prepare_input(source, out):
    source = Path(source).resolve(strict=True)
    path = source
    if source.suffix == '.zst':
        path = out / 'input.decompressed'
        with path.open('xb') as handle:
            subprocess.run(['zstd','-q','-dc',str(source)],stdout=handle,check=True)
    with path.open() as handle:
        first = next((line.strip() for line in handle if line.strip()), '')
    if not first.startswith('>'):
        validate_phylip(path)
        return path
    names = set(); lengths = []; count = 0
    for name, seq in _fasta_rows(path):
        if name is not None:
            if name in names:raise ValueError('duplicate FASTA taxon name')
            names.add(name); lengths.append(0); count += 1
        else:lengths[-1] += len(seq)
    if count < 4 or not lengths[0] or len(set(lengths)) != 1:
        raise ValueError('at least four equally long aligned sequences required')
    converted = out / 'input.phy'
    with converted.open('x') as handle:
        handle.write(f'{count} {lengths[0]}\n')
        started = False
        for name, seq in _fasta_rows(path):
            if name is not None:
                if any(c.isspace() for c in name):
                    raise ValueError('FASTA names must not contain whitespace')
                if started:handle.write('\n')
                handle.write(name + ' '); started = True
            else:handle.write(seq)
        handle.write('\n')
    return converted
