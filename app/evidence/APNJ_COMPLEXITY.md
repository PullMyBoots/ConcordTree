2026-11-07 update: Current row controls and exact-tie progress rules are documented in ../APNJ_DESIGN.md. The original raw-coordinate derivative discussion below is historical; the generic fixed-population control-variate identity still applies. Aggregate scratch is now streamed with a64MiB buffer bound.

# AP-NJ construction: cost and statistical identities

These are bounds for experimental INITIAL construction, not downstream
MLP/likelihood/MSC search, statistical consistency or dominance over DIPPER.

## Parameters and machine model

N taxa; L original aligned sites; P projections; W ordering window; K directed
neighbors; A expansion passes; R representatives; H row-sum landmarks. All
P,W,K,A,R,H are fixed independently of N,L. All-site runs retain every input
column. Fixed-word arithmetic/popcount is constant time; checked IDs/array
sizes must fit the machine domain. Labels add their own byte count B.

## Guaranteed contraction

The sparse candidate graph retains every adjacent edge of one projection
ordering, hence contains a Hamiltonian path. This path survives all capping.
In any maximal matching, unmatched vertices form an independent set on that
path and number <=ceil(m/2). With r matched edges, m-2r<=ceil(m/2), so
r>=ceil(floor(m/2)/2)=floor((m+2)/4). This remains true when greedy completion
starts from any already accepted disjoint reciprocal-best edges.

Quarter prototype: target min(m-3,floor((m+2)/4)). Then
m_next-2 <=(3/4)(m-2), hence O(log N) rounds and
sum m <=4(N-2)+2*round_count. The earlier 4N+16 shorthand is NOT universal.

Conservative variant: fixed C>=5, target ceil(m/C). For m>=4 this is feasible
by the lemma and <=m-3. Thus m_next<=(1-1/C)m, sum m<=CN and
rounds<=ceil(log(N/3)/[-log(1-1/C)]) plus an endpoint constant.
Tested conservative C=2(2PW+1) is tied to the seed-graph degree budget; this
proof uses the preserved path, not undirected ANN degrees. Large C weakens the
worst-case constant. Never apply the quarter prototype's 4N bound to it.
No empirical shrink, projection success or biological tree balance is assumed.
Forced quarter-contraction has already damaged development accuracy.

## Total work and storage

Seeding: <=PWm edges; coordinate sorting/deduplication O(m log m).
Directed width K: <=Km entries. A neighbor-of-neighbor expansion materializes
<=m(K+K^2) entries plus the path, with global sort O(mK^2 log(mK^2)). The packed
variant fixes A=2, so this remains O(m log m). The separate sketch-ANN
prototype caps passes at ceil(log2 m), but each pass has only fixed local work
O(mK^2P), also O(m log m). Do NOT combine logarithmic passes with a global
O(m log m) sort per pass and claim a single logarithm.

E representative edge queries cost O(E R^2 ceil(L/16)) word visits; <=Hm
landmark queries. Exact raw-row controls cost O(mRL+mR^2L), with O(L) aggregate
storage. Representative/sketch merges and iterative topology assembly add
O(mR+mP+m); no descendant lists. Sparse-Q sorting is O(m log m). Per round is
therefore O(mL+m log m). Geometric contraction gives O(NL+N log N) work.

Packed input uses O(NL) storage; representative/sketch/sparse state O(N),
control scratch O(L). The earlier dense-profile prototype instead retains
O(NL) floating profiles. One gamma fit uses 32 triplet scans and 48 fixed
scalar-search steps, O(L), without simulator parameters. Parsing is at least
Omega(NL); iterative export is O(N+B). Dense matrices under diagnosis/ are
validation oracles and excluded from runtime claims. Hash convenience code,
label handling, sort guarantees and device capacity still need implementation
audit; this proof concerns the explicitly bounded integer/array algorithm,
not arbitrary memory capacity, bit complexity or downstream convergence.

## Exact raw-row control statistics

For active subtree i let f_i(s,b)=sum_r w_ir 1[x_ir,s=b]. Unknowns contribute
zero to the four observed-base coordinates. Active clusters have disjoint
leaf supports. Set F=sum_i f_i. For i!=j define
p_ij=1-<f_i,f_j>/L, and p_ii=0. This is an all-site CONTROL statistic, not a
claim that unknowns are biological mismatches in the primary metric. Then

    P_i=sum_(j!=i)p_ij=(m-1)-[<f_i,F>-<f_i,f_i>]/L.

For r_i=f_i(A)+f_i(G), y_i=f_i(C)+f_i(T), R=sum_i r_i, Y=sum_i y_i,
q_ij=sum_s(r_i*y_j+y_i*r_j)/L with zero diagonal has exact row

    Q_i=sum_s[r_i*(Y-y_i)+y_i*(R-r_i)]/L.

These take O(mRL) observation work, O(L) aggregate state. Independent raw
observation checks cover missing data and nonuniform representative weights.

## Landmark control variate

Let D_ij be primary corrected representative distance. Sample H uniform active
landmarks without replacement, excluding i itself. The h_i remaining samples
are uniform among other active nodes. With exact P_i,Q_i and deterministic
coefficients beta_i,gamma_i, use

    estimated_row_i=(m-1)*mean_sample(D_ij)
      -beta_i*[(m-1)*mean_sample(p_ij)-P_i]
      -gamma_i*[(m-1)*mean_sample(q_ij)-Q_i].

Conditionally on the current representative population, ideal independent
uniform sampling makes this unbiased for sum_(j!=i)D_ij. Coefficients may
depend on exact population raw means, not fitted sampled residuals for this
argument. The prototype uses clipped JC/gamma/K2P-gamma derivatives at exact
raw means, setting saturated/clamped derivatives to zero. H>=m uses exact
rows without cancellation correction. CONTROL_VALIDATION.json exhaustively
checks subset averages in small finite populations. This does NOT guarantee
variance reduction, finite-precision bounds, topology accuracy or consistency.
Fixed H,R do not eliminate noise as N grows; rate-model mismatch remains.

## Output-shape limitation of cherry-only round contraction

A rake-only process starting from singleton leaves and merging disjoint pairs per round yields a rooted construction tree of height at most the number of rounds. A worst-case geometrically contracting schedule therefore cannot exactly reconstruct every pectinate tree: the unrooted internal path of an N-leaf caterpillar has length N-3, whereas every pair of leaves in the round-construction tree has graph distance at most twice its round count plus 2. Thus accuracy guarantees for arbitrary shapes do not follow from the complexity proof. Conservative large fixed divisors retain the formal asymptotic bound but have weak constants and only defer, not eliminate, this structural limit. Current scientific goal remains empirical superiority, not an all-tree consistency guarantee. A future universally additive-consistent route would need path-compression/insertion/separator operations in addition to cherry contraction.

Literature audit: FastTree 2009 explicitly distinguishes correction of an average from averaging corrected distances (https://doi.org/10.1093/molbev/msp077). Sparse Neighbor Joining 2024 (https://doi.org/10.1093/bioinformatics/btae701) bounds distance queries by N log N but its direct centroid enumeration adds N squared work (section3.2.2); that implementation cannot supply the requested whole-algorithm N log N guarantee. These are source attributions, not copied guarantees for this prototype.

## Fixed-size exact terminal solve (B=512)

For m<=B active clusters, form the symmetric matrix of primary corrected
representative distances (not a full unbounded N-by-N matrix), then run
standard global-minimum NJ to three nodes. The matrix omits additive node
offsets because NJ Q rankings and reductions are invariant under adding
u_i+u_j off diagonal. Finite-precision ties are not claimed to be gauge exact.
Signed reduced distances are retained. Initial rows sum in ascending slot
order and updates preserve that arithmetic. This stage is a new scientific
candidate, not an exact implementation of sparse approximate NJ.

Terminal distance work is O(B^2 R^2 L), standard scanning NJ O(B^3), matrix
storage O(B^2). Total is O(NL+N log N+B^2 R^2 L+B^3) time and
O(NL+N+B^2) space. B is fixed independently of N and L. Thus the requested
N growth remains O(N log N) for fixed L and fixed budgets, with bounded
dense work explicitly exposed. Small-N timing can be cubic up to B and
global empirical curves need not match a single asymptotic power.
The earlier cherry-only outer-round shape limitation persists for N>B.

## Bounded local cleanup candidate (four passes, four representatives)

For each directed tree edge, a two-pass message DP stores up to four nearest
leaf IDs with their integer topological distances; degree is at most three.
Every message sorts at most eight items, costing O(N) total and O(N) space.
For each internal edge, its four component summaries define six mean corrected
leaf-pair distances. There are6(N-3) component queries and at most16 leaf-pair
contributions per query, each using all L sites. Three four-point scores are
formed, current-versus-alternative gains sorted in O(N log N), and a greedy
central-edge closed-neighborhood exclusion applies compatible improving NNIs.
Four passes is a fixed unconditional cap (early exit if no moves). Thus this
adds O(NL+N log N) work and O(N) auxiliary storage. The numerical distance model
and input sites are unchanged. It is an initialization refinement, not the
previous raw sparse-NJ tree, and its entire cost must be included in comparisons.
It does not load/change MLP or the subsequent likelihood/MSC refinement.
This is an empirical robustness mechanism; no general all-model statistical
consistency, monotone true-nRF improvement or global minimum-evolution optimum
is claimed. Four-point/NNI ideas are established, not an originality claim.

## Observed-intersection follow-up

Changing the primary per-leaf-pair usable mask from observed union to observed
intersection does not change query counts, storage or any cost bound. It avoids
counting one unknown character as an evolutionary mismatch. The exact raw control
P/Q remain their original all-site zero-filled-coordinate statistics, and their
control-variate expectation identity still holds for any fixed primary D. The
current derivative coefficients may be inefficient for this different D; no
variance-reduction or general missing-data/MSC consistency theorem is asserted.
No-overlap pairs retain the explicitly defined finite saturated fallback.

## Exact within-round query reuse

Current and previous ANN pools are lexicographically sorted unique canonical
pairs, with unchanged representative arrays within that outer round. A linear
merge identifies common pairs; reuse their exact prior computed doubles and
query only newly introduced pairs, preserving orientation and reduction order.
This adds O(E_old+E_new) work and O(E_new) indexing state per fixed ANN pass;
no hash-success assumption, unbounded cache history or extra logarithm is needed.
It does not change candidates, scores or tree output. This differs from the
archived sorted global leaf-pair cache: no repeated key sorting or cross-round
representative identity bookkeeping. Its practical benefit remains a measured
engineering question; source equality alone does not prove a speed gain.

## Frozen implementation audit (2026-11-06)

The public closure fixes P=16,W=8,K=16,A=2,R=2,H=32,C=514,B=512.
One seed pool has at most128m edges. Each expanded pool has at most273m
entries before deduplication; two expanded distance-query pools plus the seed
and landmarks total at most706m queries per round. Final directed pruning
and the preserved path leave at most17m-1 candidate edges. Thus aggregate
outer query count is at most706*514*N, a deliberately loose deterministic
bound. Four cleanup sweeps add at most24(N-3) component queries of width4.
COMPLEXITY_INVARIANT_AUDIT.json checks these bounds and the exact frozen
constants against all900 archived histories; it complements the general
proof rather than proving it by finite testing.

The sparse outer rounds, bounded NJ terminal and four-pass cleanup form the
initialization algorithm covered by the theorem. For fixed L and all fixed
budgets its numeric work is O(N log N); with input length exposed it is
O(NL+N log N+B^2 R^2 L+B^3), space O(NL+N+B^2). This does not bound downstream
MLP/likelihood/MSC convergence. Public label validation uses comparison sort:
if S is maximum label length, add O(S*N log N) worst-case character work.
Newick serialization adds the actual output bytes. Numeric native tree
operations use fixed-degree adjacency arrays, not recursive descendant sets.
The Python Tree handoff uses integer dictionaries/sets; their usual amortized
lookup/allocator cost is an implementation assumption for the public wrapper,
not an unconditional arbitrary-hash bit-complexity theorem. The numeric
construction bound itself has no successful-hashing assumption.

Public dimensions are guarded before allocating native state:
4<=N<=floor(INT_MAX/273),1<=L<=INT_MAX-16. Candidate query counts and
representative counts fit signed32-bit counters; canonical pair keys and
sequence offsets use64-bit integers. Raw-row coordinate/category arithmetic
uses64-bit expressions. Sequence/scalar counts stay within L. CUDA allocation
errors fail explicitly; the domain bounds do not promise enough physical RAM
or device memory for every admitted shape. Entirely unobserved taxa are
rejected, including fake observed values in unused odd padding nibbles.

The control-variate expectation argument above assumes ideal independent
uniform landmark sampling conditional on representatives. Fixed pseudorandom
seeds are the reproducibility convention of this implementation; the argument
is not a verified statement about independence of its deterministic streams
or finite-precision unbiasedness. No parameter, seed or stream was changed
after the common-mask science freeze.
