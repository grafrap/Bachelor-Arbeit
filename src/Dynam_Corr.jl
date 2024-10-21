using ITensors
using ITensorMPS
using HDF5
using InteractiveUtils
using LinearAlgebra
# using Base.Threads

function Jackson_dampening(N::Int)
  #=
    Implements the Jackson dampening factors g_n
  =#
  g = zeros(N+1)
  πN = π/(N+1)
  for n in 1:(N+1)
    g[n] = ((N - n + 1)*cos(πN*n) + sin(πN*n)/tan(πN)) / (N + 1)
  end
  return g
end

function Chebyshev_vectors(B::MPO, H::MPO, ψ::MPS, N::Int)
  #=
    Implements the Chebyshev vectors ∣t_n⟩ = T_n(Ĥ')*Ĉ|ψ⟩
  =#
  t = Vector{MPS}(undef, N+1)
  t[1] = apply(B, ψ)
  t[2] = apply(H, t[1])

  for n in 3:N+1
    println(stderr, "Calculating Chebyshev vector $n")
    t[n] = 2*apply(H, t[n-1]) - t[n-2]
  end

  return t
end

function Chebyshev_moments(ψ::MPS, A::MPO, t::Array{MPS, 1})
  #=
    Implements the Chebyshev moments μ_n = ⟨ψ|Â*T_n(Ĥ')*B^|ψ⟩ = ⟨ψ|Â∣t_n⟩
  =#
  μ = zeros(length(t))

  for n in 1:length(t)
    μ[n] = inner(ψ', A, t[n])
  end

  return μ
end

function Chebyshev_expansion(x::Float64, N::Int)
  #=
    Implements the Chebyshev expansion for the dynamical correlator
  =# 
  T = zeros(N+1)
  T[1] = 1
  T[2] = x

  for n in 3:N+1
    T[n] = 2*x*T[n-1] - T[n-2]
  end

  return T
end
  
function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, I::MPO, E0::Float64, E1::Float64, ω::Float64, N::Int)
  #=
    Implements the dynamical spin correlator for a single frequency
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#

  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  @assert ω < W "ω must be less than W"
  @assert ω > 0 "ω must be positive"

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  ω_ = ω/a - W_ # ω', scaled ω
  H_scaled = 1/a * (H - E0*I) - W_*I

  #calculate the Jackson dampening factors
  println(stderr, "Calculating Jackson dampening factors")
  g = Jackson_dampening(N)

  # calculate the Chebyshev vectors
  println(stderr, "Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N)

  # calculate the Chebyshev moments
  println(stderr, "Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  println(stderr, "Calculating Chebyshev expansion")
  T = Chebyshev_expansion(ω_, N)
  
  χ = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N+1]))
  return χ
end


function Dynamic_corrolator(ψ::MPS, H::MPO, A::MPO, I::MPO, E0::Float64, E1::Float64, ω::Vector, N::Int)
  #=
    Implements the dynamical spin correlator for a vector of frequencies
    χ(ω) = <ψ|Â δ(ωI - Ĥ - E_0) Â|ψ>
    where δ is the Dirac delta function, ψ the ground state MPS, Â the operator A[i], and Ĥ the Hamiltonian.
  =#

  ϵ = 0.025
  W = -E0 - E1 # W* = -E0 + E1
  for i in 1:length(ω)
    @assert ω[i] < W "ω must be less than W"
    @assert ω[i] > 0 "ω must be positive"
  end

  W_ = 1 - 0.5*ϵ # W', scaled W
  a = W/(2*W_)
  H_scaled = 1/a * (H - E0*I) - W_*I

  #calculate the Jackson dampening factors
  println(stderr, "Calculating Jackson dampening factors")
  g = Jackson_dampening(N)

  # calculate the Chebyshev vectors
  println(stderr, "Calculating Chebyshev vectors")
  t = Chebyshev_vectors(A, H_scaled, ψ, N)

  # calculate the Chebyshev moments
  println(stderr, "Calculating Chebyshev moments")
  μ = Chebyshev_moments(ψ, A, t)

  # calculate the Chebyshev expansion of the scaled ω
  χ = zeros(length(ω))
  
  println(stderr, "Calculating Chebyshev expansion")
  for j in eachindex(ω)
    ω_ = ω[j]/a - W_ # ω', scaled ω
    T = Chebyshev_expansion(ω_, N)
    χ[j] = 1/(a * π * sqrt(1 - ω_^2)) * (g[1] * μ[1] + 2 * sum([g[n]*μ[n]*T[n] for n in 2:N+1]))
  end
  println(stderr, χ)
  return χ
end


function main()
# read in the Groundstate MPS
f = h5open("outputs/psin_data.h5", "r")
ψ0 = read(f, "state_1/ψ", MPS)
close(f)

N = length(ψ0)

# read in the Hamiltonian
f = h5open("outputs/H_data.h5", "r")
H = read(f, "H", MPO)
close(f)

# read in the ground state energy
fr = open("outputs/Szi_Sz_pll=0.0.txt", "r")
lines = readlines(fr)
close(fr)

energy_line = findfirst(contains("List of E:"), lines) + 1
energy_array = eval(Meta.parse(lines[energy_line]))
E0 = energy_array[1]
E1 = energy_array[2]

# read in the Sz(i)
f = h5open("outputs/operators_data.h5", "r")
# S2 = read(f, "S2", MPO)
I = read(f, "I", MPO)
Sz::Vector{MPO} = []
group = f["Sz"]
for i in 1:N
  push!(Sz, read(group, "Sz_$i", MPO))
end
close(f)

# calculate the dynamical correlator
len_ω = 41
ω = collect(range(0, 2, len_ω))[2:end]
i = 1
N = 30
χ = zeros(length(Sz), len_ω-1)
# TODO: test with results from the paper: formula 9 and fig 5
# Take J = 1, J2 = 0.19J, ΔJ = 0.03J, N_sites = 18, S = 1, Sz = 1
# mpiexecjl -n 4 julia DMRG_template_pll_Energyextrema.jl 1 18 1.0 0 0 true true > outputs/output.txt 2> outputs/error.txt
#=@threads =#for i in eachindex(Sz)
  println(stderr, "Calculating χ for Sz[$i]")
  χ[i, :] = Dynamic_corrolator(ψ0, H, Sz[i], I, E0, E1, ω, N)
end
# χ = Dynamic_corrolator(ψ0, H, Sz[i], I, E0, E1, ω, N)
# println("χ(ω = $ω) = $χ")
println(χ)
end

main()

#=
how many values are in this array? 
[0.003181899443857412 0.0029570881807723048 0.002777163588597454 0.002630238166769789 0.0025085495689110763 0.0024067703449539156 0.0023210927617313714 0.0022487012010764616 0.0021874522006054974 0.0021356721481611382 0.00209202481186593 0.0020554219793302254 0.0020249616145412304 0.001999884096124892 0.0019795406406088833 0.0019633701216824426 0.0019508817896464797 0.0019416422104681212 0.001935265270252728 0.0019314044382652915 0.001929746715303284 0.0019300078542383115 0.0019319285509028158 0.00193527138214684 0.001939818324196182 0.0019453687252584432 0.001951737636255679 0.0019587544257454537 0.001966261621700476 0.001974113935362798 0.0019821774319501557 0.001990328820336931 0.0019984548395164297 0.0020064517240820914 0.0020142247344413946 0.0020216877402198254 0.002028762847491553 0.002035380062211919 0.0020414769836228906 0.0020469985225270407;
 0.0031351220268018925 0.002922166054788557 0.002748093123083234 0.0026025389115784925 0.002478814608326553 0.0023723853816379633 0.002280050008468901 0.0021994700740731437 0.0021288858664864686 0.002066937705635837 0.002012549636383601 0.0019648514758593668 0.0019231252470376006 0.0018867675690829179 0.0018552627530458552 0.0018281632389625 0.001805075165831018 0.0017856475924527843 0.0017695643550087368 0.0017565378551056036 0.001746304278562239 0.0017386198862225084 0.0017332581159154928 0.001730007303560326 0.0017286688805586057 0.0017290559401191424 0.001730992091100193 0.0017343105371089792 0.001738853332880579 0.0017444707807049323 0.001751020937825374 0.0017583692119677106 0.0017663880269636314 0.0017749565441578378 0.001783960428197177 0.0017932916480831683 0.001802848306172846 0.001812534489242775 0.0018222601368727705 0.0018319409233187014;
 0.003143656859016429 0.0029270974275206873 0.0027510749812415377 0.0026048170098305454 0.002481346955272673 0.0023759173408864557 0.0022851639472218784 0.002206619781613941 0.0021384213737843647 0.002079123788556 0.0020275800080071064 0.001982859942932438 0.00194419466928563 0.0019109371897075033 0.0018825342956163853 0.001858506051916664 0.0018384306188190952 0.0018219328756019825 0.001808675794730751 0.0017983538331991695 0.0017906878217646123 0.0017854209788536392 0.0017823157773684346 0.001781151464113182 0.0017817220826186787 0.0017838348870621195 0.0017873090619841558 0.0017919746824658776 0.0017976718643291274 0.0018042500651483315 0.001811567505389213 0.0018194906855198397 0.001827893979975993 0.0018366592927752464 0.0018456757626320488 0.0018548395078305767 0.0018640534030142366 0.0018732268815615488 0.00188227575842603 0.0018911220692861773;
 0.003138922819990419 0.0029232618286673968 0.002747411960509627 0.002600796113003642 0.002476576378089492 0.0023701080622857053 0.002278105691736747 0.002198164127396543 0.0021284694106354154 0.0020676168031463177 0.0020144922594549893 0.0019681929287341917 0.0019279724838003138 0.0018932027033969465 0.0018633459643556457 0.0018379352194199878 0.0018165592117330388 0.0017988514161897731 0.0017844816740683193 0.0017731498008063784 0.0017645806571477774 0.0017585203175625194 0.001754733069562614 0.001752999047765483 0.001753112356687703 0.0017548795724800777 0.001758118540297499 0.0017626574035592931 0.0017683338159498374 0.0017749942979938327 0.0017824937083811495 0.0017906948065959422 0.0017994678883232736 0.0018086904789227614 0.0018182470732387116 0.001828028912358149 0.0018379337897772028 0.0018478658809050427 0.00185773559100626 0.0018674594176212222;
 0.0031396619367899353 0.002923585306051055 0.0027474833525790577 0.002600740054226679 0.002476490142585236 0.00237006873015486 0.0022781748583182936 0.0021983911481048397 0.002128893726859826 0.0020682696663351353 0.0020153980465806064 0.0019693701715859414 0.0019294346923832862 0.0018949590376380198 0.0018654017916623798 0.0018402925846979553 0.0018192172390984302 0.0018018066565396798 0.0017877284091067276 0.0017766803115738736 0.0017683854632498898 0.0017625883919235412 0.0017590520325095921 0.0017575553434721065 0.0017578914144151516 0.001759865954592752 0.0017632960786727345 0.0017680093257274867 0.0017738428620763201 0.001780642829633877 0.0017882638097935796 0.0017965683792829322 0.0018054267393672182 0.0018147164036111953 0.0018243219324028876 0.001834134704795831 0.001844052720084611 0.0018539804230043972 0.001863828547622896 0.00187351397593704;
 0.0031389802251794 0.0029229957078111277 0.0027468993125546763 0.002600098765700591 0.0024757456545578573 0.002369187584206845 0.0022771331592731942 0.002197172501833167 0.0021274877309741167 0.00206667076927411 0.002013604661481431 0.0019673839721217677 0.001927260042236019 0.0018926025205478797 0.0018628718188339864 0.0018375990628640888 0.0018163712869771528 0.0017988203605049777 0.0017846146110825848 0.0017734524237455558 0.0017650573053503902 0.0017591740477236207 0.0017555657227978935 0.001754011313317929 0.0017543038328984768 0.0017562488254946446 0.0017596631608637902 0.001764374062187521 0.001770218316637479 0.0017770416306676748 0.0017846981001691708 0.0017930497720111411 0.0018019662784184356 0.0018113245294564832 0.001821008451879702 0.0018309087649437672 0.0018409227856341637 0.0018509542572343804 0.0018609131963299578 0.0018707157542848403
 0.00313901498033341 0.002922978289291042 0.002746845922363976 0.0026000220341879272 0.0024756557132826493 0.0023690927235993585 0.0022770402556147383 0.002197087307791902 0.0021274150812494994 0.0020666147311728376 0.0020135686484959677 0.0019673708318955875 0.0019272721263312576 0.001892641740986148 0.0018629396948267824 0.001837696760133112 0.0018164996513840572 0.001798979947243421 0.0017848057103603283 0.001773675083652188 0.0017653113524396884 0.0017594591057168338 0.0017558812297234143 0.0017543565373345036 0.0017546778869975467 0.0017566506812373955 0.0017600916612779288 0.001764827933924964 0.0017706961814748887 0.001777542016417421 0.0017852194510559068 0.00179359045856011 0.0018025246068936014 0.001811898750880701 0.0018215967706639274 0.001831509347148146 0.001841533766880672 0.0018515737502878653 0.0018615392983619995 0.0018713465538331493;
 0.003138918823375111 0.002922890693912294 0.002746756093568569 0.002599922215323071 0.0024755403067670896 0.0023689577326194873 0.002276882908117191 0.0021969057909702484 0.0021272083466513827 0.0020663823469814556 0.0020133106845272716 0.0019670877680265982 0.0019269647781861043 0.0018923111986493136 0.0018625872716058344 0.0018373239492249689 0.0018161080888647776 0.0017985713801968815 0.0017843819692405726 0.0017732380581784329 0.0017648629704719683 0.001759001314672542 0.0017554159801799472 0.00175388576852841 0.0017542035139817237 0.001756174583497314 0.0017596156726401293 0.0017643538336165074 0.0017702256862128461 0.0017770767734247386 0.0017847610319135301 0.0017931403538171931 0.0018020842213668615 0.0018114693995824624 0.0018211796753049069 0.001831105633167257 0.0018411444609588085 0.0018511997783072114 0.0018611814837759101 0.0018710056164151585;
 0.003138915708788867 0.0029228814410022065 0.0027467419753900383 0.002599904275719591 0.0024755194317985954 0.0023689346928468416 0.0022768583850030434 0.0021968803940924132 0.0021271826253454087 0.0020663567984758554 0.00201328575980137 0.0019670638761528966 0.001926942289643301 0.001892290447924899 0.0018625685593017017 0.0018373075438120428 0.0018160942281881132 0.0017985602728019206 0.001784373795571461 0.0017732329716843383 0.0017648610986473845 0.00175900276005048 0.001755420821291374 0.0017538940608376736 0.001754215290804871 0.0017561898569081749 0.0017596344343674692 0.001764376055934321 0.0017702513228251991 0.0017771057603458234 0.0017847932883432228 0.001793175783011863 0.001802122711505153 0.0018115108246255748 0.0018212238958505334 0.001831152497296588 0.0018411938050758795 0.001851251427969499 0.001861235254515643 0.0018710613145504246;
 0.003138915711690678 0.002922881444211611 0.002746741978769736 0.0025999042791605104 0.002475519435212205 0.002368934696160225 0.002276858388155677 0.002196880397033813 0.002127182628033506 0.002066356800875739 0.0020132857618843725 0.0019670638778958058 0.0019269422910277803 0.0018922904489369053 0.001862568559931111 0.0018373075440522564 0.0018160942280357286 0.0017985602722564594 0.001784373794635085 0.0017732329703616342 0.0017648610969451587 0.0017590027579775548 0.0017554208188583513 0.001753894058056863 0.001754215287690055 0.0017561898534744727 0.0017596344306312333 0.0017643760519129553 0.001770251318537067 0.0017771057558100827 0.0017847932835797567 0.0017931757780411145 0.0018021227063481128 0.0018115108193035518 0.0018212238903851687 0.0018311524917097046 0.0018411937993893976 0.001851251422205369 0.0018612352486957633 0.0018710613086965592;
 0.0031389188241575825 0.0029228906933816846 0.0027467560922404682 0.002599922213594376 0.0024755403049507726 0.002368957730966424 0.002276882906830294 0.0021969057902143175 0.0021272083465600296 0.002066382347662393 0.002013310686066266 0.0019670877704905695 0.001926964781625584 0.0018923112031005575 0.001862587277092496 0.001837323955759536 0.0018161080964498408 0.0017985713888261645 0.0017843819788999405 0.00177323806884674 0.001764862982121748 0.0017590013272708069 0.0017554159936887539 0.0017538857829054795 0.0017542035291809154 0.001756174599469181 0.0017596156893323607 0.0017643538509743632 0.0017702257041795875 0.0017770767919419307 0.0017847610509214868 0.0017931403732552643 0.001802084241173792 0.0018114694196966377 0.0018211796956646954 0.001831105653711219 0.0018411444816259918 0.0018511997990373979 0.0018611815045098104 0.001871005637094626;
 0.0031390149963942725 0.0029229783038880034 0.0027468459357875585 0.0026000220466584644 0.0024756557249743932 0.0023690927346544607 0.0022770402661521894 0.002197087317913028 0.0021274150910419753 0.0020666147407132876 0.002013568657851838 0.001967370841126663 0.0019272721354906648 0.0018926417501211208 0.001862939703979375 0.0018376967693407057 0.0018164996606796684 0.0017989799566562141 0.0017848057199158045 0.0017736750933724276 0.001765311362343648 0.0017594591158205147 0.0017558812400400566 0.0017543565478747311 0.001754677897769517 0.001756650692247007 0.0017600916725289248 0.001764827945419051 0.0017706961932119052 0.001777542028395416 0.0017852194632712991 0.0017935904710077897 0.001802524619567037 0.00181189876377207 0.0018215967837642048 0.0018315093604472181 0.0018415337803674567 0.0018515737639503828 0.0018615393121874724 0.0018713465678081528;
 0.0031389802137106446 0.0029229956968823026 0.002746899302023595 0.002600098755470774 0.0024757456445629695 0.0023691875744013308 0.002277133149626266 0.002197172492324706 0.0021274877215919814 0.0020666707600119887 0.002013604652337336 0.0019673839630969837 0.0019272600333341834 0.0018926025117742618 0.0018628718101950368 0.0018375990543669286 0.0018163712786292744 0.0017988203523139983 0.0017846146030559675 0.0017734524158904655 0.0017650572976735184 0.0017591740402311655 0.0017555657154953682 0.001754011306210145 0.0017543038259894847 0.001756248818787727 0.0017596631543613723 0.0017643740558912653 0.001770218310548161 0.0017770416247853114 0.0017846980944929269 0.001793049766539415 0.0018019662731488626 0.0018113245243859591 0.0018210084470044305 0.0018309087602592822 0.0018409227811353528 0.0018509542529155571 0.0018609131921848608 0.0018707157503067044;
 0.0031396619729119933 0.002923585340076504 0.0027474833848566698 0.0026007400850094793 0.0024764901720655966 0.002370068758484834 0.002278174885622115 0.0021983911744868595 0.002128893752409715 0.0020682696911314847 0.0020153980706934392 0.001969370195078416 0.001929434715312948 0.0018949590600577818 0.0018654018136212338 0.0018402926062414723 0.0018192172602691564 0.0018018066773774453 0.0017877284296488945 0.0017766803318555676 0.001768385483304085 0.0017625884117812472 0.0017590520521999625 0.0017575553630224724 0.0017578914338511895 0.0017598659739384995 0.0017632960979506735 0.0017680093449585967 0.0017738428812801512 0.0017806428488285875 0.0017882638289959884 0.0017965683985086257 0.0018054267586304993 0.0018147164229252268 0.0018243219517797232 0.0018341347242464396 0.001844052739618932 0.0018539804426314153 0.0018638285673506921 0.0018735139957728076;
 0.003138922840238199 0.002923261848935632 0.0027474119805712704 0.0026007961326727272 0.002476576397212588 0.0023701080807363275 0.0022781057094108568 0.0021981641442092403 0.002128469426518453 0.0020676168180459786 0.0020144922733303064 0.001968192941555675 0.001927972495548536 0.0018932027140616272 0.00186334597393468 0.0018379352279186251 0.0018165592191631828 0.0017988514225692925 0.001784481679420458 0.0017731498051592816 0.0017645806605338737 0.0017585203200181283 0.0017547330711275503 0.0017529990484825531 0.001753112356602442 0.0017548795716403464 0.0017581185387531637 0.0017626574013619336 0.0017683338131524319 0.0017749942946505258 0.0017824937045469514 0.0017906948023265119 0.0017994678836746984 0.001808690473951303 0.001818247068000648 0.0018280289069095349 0.0018379337841737294 0.0018478658752018433 0.0018577355852577538 0.0018674594118809535;
 0.0031436569242337784 0.0029270974895114054 0.0027510750403819365 0.002604817066379107 0.0024813470094175423 0.0023759173927725835 0.0022851639969668574 0.002206619829318032 0.0021384214195368204 0.002079123832439088 0.0020275800500988658 0.001982859983308311 0.0019441947080198125 0.0019109372268735021 0.0018825343312875828 0.0018585060861665151 0.0018384306517212922 0.001821932907230491 0.001808675825159707 0.0017983538625028477 0.0017906878500173163 0.0017854210061295614 0.0017823158037414638 0.001781151489656813 0.0017817221074058103 0.0017838349111649115 0.0017873090854738846 0.001791974705412752 0.0017976718868022014 0.0018042500872153097 0.0018115675271163831 0.0018194907069718732 0.0018278940012158935 0.0018366593138642224 0.001845675783629381 0.0018548395287936068 0.0018640534239982023 0.001873226902619544 0.0018822757796089387 0.0018911220906426011;
 0.003135122112773274 0.0029221661337877924 0.0027480931969147255 0.0026025389815744525 0.0024788146755018825 0.0023723854467836603 0.0022800500722113897 0.002199470136913986 0.002128885928830061 0.0020669377678090885 0.0020125496986505755 0.0019648515384322845 0.0019231253100854993 0.001886767632738416 0.0018552628174104626 0.0018281633041111302 0.001805075231815641 0.0017856476593054906 0.0017695644227443746 0.001756537923723984 0.0017463043480501275 0.0017386199565552708 0.0017332581870586313 0.0017300073754708124 0.0017286689531860508 0.0017290560134069028 0.001730992164986314 0.0017343106115270555 0.001738853407760605 0.0017444708559740227 0.0017510210134084581 0.0017583692877881278 0.0017663881029437098 0.0017749566202194503 0.0017839605042622268 0.0017932917240740802 0.0018028483820129088 0.0018125345648566781 0.0018222602121868304 0.001831940998261314;
 0.0031818996926240295 0.002957088410521405 0.0027771638038390886 0.0026302383708278853 0.0025085497643197847 0.0024067705336947407 0.002321092945383184 0.0022487013809149308 0.002187452377671956 0.002135672323311111 0.00209202498580505 0.002055422152641046 0.0020249617877038054 0.0019998842695328925 0.0019795408145823956 0.001963370296478367 0.0019508819654670633 0.0019416423874680506 0.0019352654485450846 0.0019314046179267008 0.0019297468963782813 0.0019300080367431523 0.0019319287348289207 0.0019352715674636803 0.0019398185108539575 0.0019453689131905514 0.001951737825380739 0.001958754615969367 0.0019662618129182075 0.0019741141274599865 0.0019821776248046784 0.00199032901382029 0.001998455033495029 0.0020064519184185042 0.002014224928995427 0.0020216879348496516 0.002028763042054648 0.002035380256566034 0.0020414771776268805 0.0020469987160417094]
 =#