"""ConcordTree V1.0.2: unified sequence-to-gene/species-tree entry point."""
import argparse
from pathlib import Path
import sys

# Explicit local imports also work with Python -I, independent of cwd/PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))

def main():
    parser=argparse.ArgumentParser(description='Shared full-site bounded AP-NJ + MLP-NNI, followed by task-specific refinement.')
    parser.add_argument('--version',action='version',version='ConcordTree 1.0.2')
    parser.add_argument('--task',choices=['gene','species'],required=True)
    parser.add_argument('--input',type=Path,required=True,help='aligned DNA in PHYLIP or FASTA, optionally .zst')
    parser.add_argument('--out',type=Path,required=True,help='fresh project-owned output directory outside data/')
    parser.add_argument('--sample-fraction',type=float,help='legacy compatibility option; AP-NJ initialization always uses all sites')
    parser.add_argument('--missing-data-model',choices=['standard','coverage-aware'],default='standard',
                        help='compatibility option; initialization uses observed-intersection distances in both modes')
    parser.add_argument('--stop-ratio',type=float,default=.005,help='task-specific posterior stopping ratio; MLP uses .005')
    parser.add_argument('--model',help='gene likelihood model; default GTR+FO+I+G4')
    parser.add_argument('--views',type=int,default=1,help='gene initialization views; species requires 1')
    parser.add_argument('--rounds',type=int,help='gene posterior sweep budget; excludes explicit --stop-ratio')
    parser.add_argument('--loci',type=Path,help='species only: observed locus JSONL manifest')
    parser.add_argument('--gpu',type=int,default=0,help='physical CUDA GPU index')
    parser.add_argument('--cpu-threads',type=int,default=16)
    parser.add_argument('--no-support',action='store_true',help='omit lightweight final-branch evidence annotations')
    args=parser.parse_args()
    import sys
    if args.rounds is not None and any(x=='--stop-ratio' or x.startswith('--stop-ratio=') for x in sys.argv[1:]):
        parser.error('--rounds and --stop-ratio are mutually exclusive')
    if args.task=='gene' and args.loci is not None:parser.error('--loci is for species trees only')
    if args.task=='species' and (args.model is not None or args.views!=1 or args.rounds is not None):
        parser.error('--model, multiple --views, and --rounds are for gene trees only')
    if not args.input.is_file():parser.error('input file does not exist')
    if args.loci is not None and not args.loci.is_file():parser.error('locus manifest does not exist')
    from engine import UnifiedEngine
    try:
        result=UnifiedEngine(gpu=args.gpu,cpu_threads=args.cpu_threads).infer(
            args.input,args.out,task=args.task,model=args.model,loci=args.loci,
            sample_fraction=args.sample_fraction,views=args.views,max_rounds=args.rounds,
            stop_ratio=args.stop_ratio,missing_data_model=args.missing_data_model,branch_support=not args.no_support)
    except (ValueError,FileNotFoundError) as exc:parser.error(str(exc))
    print(result['prediction'])

if __name__=='__main__':main()
