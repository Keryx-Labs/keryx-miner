# PoC de Posse por Fatia (Shard) para o PoM ("network-fatia")

Status: **PROVA DE CONCEITO — incompleta, exploratória.** Não ativada na Mainnet nem na
Testnet. Só foi exercitada numa Simnet privada e isolada, criada especificamente para este
trabalho. Nada nesta branch altera o consenso de nenhuma rede compartilhada;
`shard_poc_activation` é `never()` em todo lugar, exceto no `SIMNET_PARAMS` privado adicionado
aqui.
Branch: `shard-poc` tanto em `keryx-miner` quanto em `keryx-node`.
Repositório companheiro: este documento é duplicado, na íntegra, em `keryx-node` e em
`keryx-miner` (as duas metades do PoC vivem em repositórios separados e precisam ser lidas
juntas).

## 1. Motivação

Hoje, os tiers de PoM exigem que uma única GPU segure e percorra (walk) o modelo inteiro. À
medida que os modelos servidos crescem (ex.: um tier de 48B parâmetros), isso deixa de caber
numa única GPU de consumidor. Este PoC explora uma alternativa: dividir um modelo em **fatias
(shards)** de tamanho fixo em bytes, ao longo de faixas de camadas (layers), e deixar que cada
GPU prove posse apenas da sua própria fatia, em vez do modelo inteiro — assim, um modelo grande
demais para qualquer placa isolada ainda pode ser minerado, cooperativamente, por várias placas.

Isso é uma questão de prova de posse, não de servir inferência: **não** se tenta tornar um
dispositivo de fatia capaz de rodar inferência OPoI real sobre o modelo completo (ver §6,
"O que não foi feito").

## 2. O que mudou — lado do miner (`keryx-miner`, branch `shard-poc`)

- `src/shard.rs`: `ShardManifest { index: u16, layer_lo: u32, layer_hi: u32, tensor_names:
  Vec<String> }` e `pack_shards(meta, target_bytes) -> Vec<ShardManifest>` — divide os tensores
  de um GGUF em fatias por faixa de camadas, cada uma próxima do `target_bytes` em disco.
- `src/pom.rs`: `WeightIndex::build_from_gguf_subset(...)` — constrói um índice de chunks no
  host (a mesma estrutura estilo Merkle usada num índice de tier inteiro), mas só sobre o
  subconjunto de tensores de uma fatia.
- `src/pom_gpu.rs`:
  - `load_raw_subset(...)` — upload bruto pra GPU restrito ao subconjunto de tensores de uma
    fatia. Dispositivos de fatia nunca rodam o motor llama e nunca fazem zero-dup contra uma
    árvore residente do modelo inteiro — é sempre um upload bruto e escopado (raw scoped
    upload).
  - Registro por dispositivo de qual fatia ele minera (`set_shard_for_device` /
    `shard_for_device`) e `ensure_shard_installed_inner`, que contém o **N-guard**: depois de
    instalar, compara a contagem de chunks obtida pela GPU contra a contagem do índice do host e
    recusa instalar um miner (`"gather N != shard index N — refusing to mine"`) em qualquer
    divergência, em vez de produzir silenciosamente uma prova sobre os dados errados. Verificado
    ao vivo em hardware real — ver §5.
- `src/models.rs`: `pom_shard_tier_index(model_id, target_bytes, shard_index, daa)` — mapeia uma
  atribuição de fatia pra um índice de tier de PoM. Neste PoC: `5 + shard_index` pra uma divisão
  em 4 fatias de 8 GB do tier `very-high` (tiers 5, 6, 7, 8). Esse deslocamento é uma
  **convenção mantida manualmente**, não negociada por nenhum protocolo — ver §6.
- `src/cli.rs` / `src/main.rs`: `--shard TIER:TARGET_GIB:IDX[,...]` (minerar uma fatia de
  tamanho fixo em vez do tier inteiro) e `--print-shards TIER:TARGET_GIB` (calcular e imprimir o
  manifest de uma fatia, depois sair — usado pra produzir os valores exatos colados na tabela de
  fatias do node; ver §3).
- Dois defeitos reais encontrados e corrigidos rodando isso ao vivo (`src/client/grpc.rs`):
  1. O gate existente "OPoI é obrigatório: recusar minerar sem nenhum modelo carregado" não
     tinha isenção para dispositivos que só minam fatia, que por design nunca carregam o modelo
     inteiro. Corrigido com `pom_gpu::any_shard_devices_active()` e uma isenção do gate para
     processos que só rodam dispositivos de fatia — nenhuma mudança para a mineração normal de
     tier inteiro.
  2. Uma chain com produtor único / ociosa (a Simnet deste PoC, com zero peers) nunca voltava a
     pedir um template de bloco fresco fora de um ciclo de inferência em andamento, então a
     mineração travava indefinidamente na DAA 0. Corrigido com um re-poll de fallback
     incondicional de ~500ms.
- `examples/shard_walk_poc.rs` e `examples/shard_guard_poc.rs` — ver §5.

## 3. O que mudou — lado do node (`keryx-node`, branch `shard-poc`)

- `consensus/core/src/config/params.rs`:
  - `POM_SHARDS_POC`: uma tabela de tiers de fatia (`crate::pom::PomTier { model_id, root,
    chunks }` por fatia) — valores reais, medidos, colados literalmente da saída de
    `keryx-miner --print-shards very-high:8` rodado contra o arquivo de modelo real (ver §4).
    Protegida atrás de `shard_poc_activation`.
  - `shard_poc_activation`: um novo gate de ativação de fork. `never()` no `MAINNET_PARAMS` e no
    `TESTNET_PARAMS` público compartilhado — este PoC nunca encosta em nenhum dos dois. Definido
    como `new(1)` só no `SIMNET_PARAMS`, junto com um perfil de ativação PoM/OPoI/série-H
    completo e internamente consistente, construído especificamente para uma Simnet privada
    partindo do genesis (o raciocínio de cada campo está documentado inline no diff, incluindo
    uma divergência corrigida contra os valores de `pom_v4_activation`/`h10_activation` do
    `TESTNET_PARAMS` público — o binário real do miner ativa esses gates na DAA 500 sob
    `--testnet`, não na DAA 1 como o `TESTNET_PARAMS` diz atualmente; esse desvio pré-existente
    está dormente na Testnet real só porque seu tip já está bem além da DAA 500, e não foi
    copiado pro `SIMNET_PARAMS`).

## 4. Ambiente usado na validação

- Hardware: 2 GPUs NVIDIA na mesma máquina — ordinal CUDA 0 = RTX 3090 (24 GB), ordinal CUDA 1 =
  RTX 4090 (24 GB). Nesta máquina, o ordinal do driver CUDA não bate com o índice do
  `nvidia-smi` por padrão; a validação usou `CUDA_DEVICE_ORDER=PCI_BUS_ID` pra fixar isso.
- Modelo: Kimi-Linear-48B, quantização Q4_K_M, ~29,7 GB de GGUF em disco — grande demais pra
  caber inteiro numa única placa desta máquina.
- Divisão: alvo de 8 GB por fatia → 4 fatias. Manifest real (capturado via `--print-shards
  very-high:8`):

  | Fatia | Camadas | Chunks | Raiz Merkle |
  |---|---|---|---|
  | 0 | [0, 7) | 219.852.216 | `a73a38fe1ef0cddc6e108eca25eb1de1adcdc7a65c01f87058530ac06df669b4` |
  | 1 | [7, 14) | 238.829.076 | `db22407efc04f5d261498afc7acb397c87e692699484a9bcf2a747636adda9e5` |
  | 2 | [14, 21) | 243.638.100 | `9c3aed4cc240baea0928f8282ea91ce6b0cd275f8286c45c440017c457f8cb8d` |
  | 3 | [21, 27) | 225.674.672 | `233fdb1f9b551bb7895a0fca931ce5eddd16d87a12980f7e4361e020cfe29ff2` |

- Rede: uma instância Simnet privada e isolada (sem DNS seeders, genesis próprio) — nunca a
  Testnet ou a Mainnet públicas e compartilhadas.
- Build: Windows, toolchain MSVC via Visual Studio Developer Shell; `KERYX_LLAMA_SKIP=1` (pula
  só o build da biblioteca CMake do llama.cpp usada pra servir inferência OPoI — irrelevante pro
  caminho do kernel CUDA de PoM que este PoC exercita).

## 5. Resultados — comprovados em hardware real

1. **Prova de conceito em processo único** (`keryx-miner/examples/shard_walk_poc.rs`): cada GPU,
   independentemente, empacotou o manifest da sua fatia, instalou (upload bruto e escopado
   real), rodou um walk de PoM real na GPU contra um alvo sintético (trivialmente fácil),
   construiu uma testemunha `PomProofV3` real a partir do walk, e se auto-verificou com a mesma
   função `verify_proof_v3` que um node usa — sem precisar de node ao vivo nesta etapa. As duas
   se auto-verificaram como `true`.
2. **Aceitação real na rede** (loop de mineração gRPC real, contra o node Simnet privado rodando
   o perfil de `SIMNET_PARAMS` do §3): o node genuinamente aceitou blocos de PoM em nível de
   fatia pelo caminho real de submissão de blocos, de forma sustentada, com zero rejeições — 817
   blocos na fatia 0 (GPU0/3090, ~1,05M h/s, kernel `v4 tensor-core`) e 413 blocos na fatia 1
   (GPU1/4090, ~1,68M h/s), 1.230 no total combinado, 0 rejeitados.
3. **Checagem de regressão do guard** (`keryx-miner/examples/shard_guard_poc.rs`): instala uma
   fatia com o manifest correto (funciona), depois tenta reinstalar após remover um nome de
   tensor do manifest — o índice do host, já em cache, continua refletindo a contagem de chunks
   correta (maior), então a contagem do segundo gather diverge e o N-guard corretamente recusa
   instalar um miner, confirmando que um manifest corrompido ou errado não consegue produzir
   silenciosamente uma prova aceitável.

## 6. O que explicitamente NÃO foi feito — em aberto, incompleto por design

Isto é uma prova de conceito, não uma funcionalidade terminada. Em particular:

- **Nenhuma inferência OPoI distribuída.** Um dispositivo de fatia ainda não consegue, por si
  só, servir inferência real sobre o modelo completo — a isenção do gate de mineração
  obrigatória por OPoI (§2) só deixa um processo que só roda fatias pular essa exigência pra
  *minerar*; ela não faz a inferência funcionar. Um pipeline real estilo ggml-rpc entre fatias
  (já escopado separadamente como spec §3.3–3.5) é trabalho futuro, fora deste PoC.
- **O bloco de bootstrap pré-PoM ainda precisa do modelo inteiro.** O primeiríssimo bloco de uma
  chain nova (abaixo de `pom_activation`) precisa ser minerado pelo caminho legado
  kHeavyHash+OPoI, que exige um dispositivo segurando o modelo *inteiro* — uma máquina que só
  tenha dispositivos de fatia não tem, sozinha, caminho pra passar desse bloco.
- **O deslocamento fatia→tier (`5 + shard_index`) é uma convenção mantida manualmente**, não
  negociada por nenhum protocolo. A ordem das linhas de `POM_SHARDS_POC` no node precisa bater
  com a numeração de fatias do miner por acordo, não por nenhum mecanismo on-chain. Uma
  divergência aqui fica silenciosamente errada, não é rejeitada — essa é uma fraqueza real que um
  design de produção precisaria fechar.
- **Não testado além desta configuração exata**: 2 GPUs, 4 fatias, um modelo
  (Kimi-Linear-48B), uma máquina. O comportamento com mais fatias, mais dispositivos, um modelo
  diferente, ou fatiamento entre máquinas (em vez de entre GPUs na mesma máquina) não foi
  explorado.
- **Nenhum design de recompensa/economia.** A posse parcial de um miner que só minera fatia hoje
  simplesmente reaproveita, posicionalmente, a tabela de recompensa comum baseada em tier. Se
  isso é justo ou sólido do ponto de vista da teoria dos jogos, comparado à posse de um miner de
  tier inteiro, não foi avaliado.
- **A ativação em `SIMNET_PARAMS` é deliberadamente exclusiva da Simnet.** Estender qualquer
  parte disso pra uma rede compartilhada (Testnet ou Mainnet) é uma decisão separada e bem
  maior — governança, tempo de lançamento e compatibilidade retroativa estão todos fora do
  escopo aqui.

## 7. Onde olhar / como reproduzir

- `keryx-node`, branch `shard-poc`: commit `76aa3c6f` (tabela de tiers de fatia + esqueleto do
  gate, com uma linha placeholder), commit `ecb0f09e` (dados reais do manifest + o perfil de
  ativação do `SIMNET_PARAMS` do §3).
- `keryx-miner`, branch `shard-poc`: commits `3aa1598` (registro de fatia + caminho de
  instalação), `177a33a` (CLI `--shard`/`--print-shards`), `79099bd` (integração na atribuição de
  tiers e no loop de mineração), `fba09c8` (isenção do gate de OPoI pra dispositivos de fatia),
  `af7ebee` (correção do re-poll de template em chain ociosa), `7136086` (os dois exemplos de
  diagnóstico do §5).
- Pra rodar de novo a prova em processo único ou a checagem do guard: `cargo run --example
  shard_walk_poc --features cuda` / `cargo run --example shard_guard_poc --features cuda` a
  partir do repositório do miner (ambos aceitam as flags `--gguf-path`, `--target-bytes` e
  `--mainnet`; usam por padrão o caminho do Kimi-Linear-48B e os gates de DAA estilo testnet
  usados acima).
