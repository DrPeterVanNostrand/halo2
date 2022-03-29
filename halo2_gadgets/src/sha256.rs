//! Gadget and chips for the [SHA-256] hash function.
//!
//! [SHA-256]: https://tools.ietf.org/html/rfc6234

use std::cmp::min;
use std::convert::TryInto;
use std::fmt;
use std::iter;

use halo2_proofs::{
    arithmetic::FieldExt,
    circuit::{Chip, Layouter},
    pasta::pallas,
    plonk::{Advice, Column, ConstraintSystem, Error},
};

mod table16;

pub use table16::{AssignedBits, BlockWord, Table16Chip, Table16Config};

use table16::{CompressionConfig, MessageScheduleConfig, SpreadTableChip, SpreadTableConfig, State, IV};

/// The size of a SHA-256 block, in 32-bit words.
pub const BLOCK_SIZE: usize = 16;
/// The size of a SHA-256 digest, in 32-bit words.
pub const DIGEST_SIZE: usize = 8;

/// The set of circuit instructions required to use the [`Sha256`] gadget.
pub trait Sha256Instructions<F: FieldExt>: Chip<F> {
    /// Variable representing the SHA-256 internal state.
    type State: Clone + fmt::Debug;
    /// Variable representing a 32-bit word of the input block to the SHA-256 compression
    /// function.
    type BlockWord: Copy + fmt::Debug + Default;

    /// Places the SHA-256 IV in the circuit, returning the initial state variable.
    fn initialization_vector(&self, layouter: &mut impl Layouter<F>) -> Result<Self::State, Error>;

    /// Creates an initial state from the output state of a previous block
    fn initialization(
        &self,
        layouter: &mut impl Layouter<F>,
        init_state: &Self::State,
    ) -> Result<Self::State, Error>;

    /// Starting from the given initialized state, processes a block of input and returns the
    /// final state.
    fn compress(
        &self,
        layouter: &mut impl Layouter<F>,
        initialized_state: &Self::State,
        input: [Self::BlockWord; BLOCK_SIZE],
    ) -> Result<Self::State, Error>;

    /// Converts the given state into a message digest.
    fn digest(
        &self,
        layouter: &mut impl Layouter<F>,
        state: &Self::State,
    ) -> Result<[Self::BlockWord; DIGEST_SIZE], Error>;
}

/// The output of a SHA-256 circuit invocation.
#[derive(Debug)]
pub struct Sha256Digest<BlockWord>([BlockWord; DIGEST_SIZE]);

/// A gadget that constrains a SHA-256 invocation. It supports input at a granularity of
/// 32 bits.
#[derive(Debug)]
pub struct Sha256<F: FieldExt, CS: Sha256Instructions<F>> {
    chip: CS,
    state: CS::State,
    cur_block: Vec<CS::BlockWord>,
    length: usize,
}

impl<F: FieldExt, Sha256Chip: Sha256Instructions<F>> Sha256<F, Sha256Chip> {
    /// Create a new hasher instance.
    pub fn new(chip: Sha256Chip, mut layouter: impl Layouter<F>) -> Result<Self, Error> {
        let state = chip.initialization_vector(&mut layouter)?;
        Ok(Sha256 {
            chip,
            state,
            cur_block: Vec::with_capacity(BLOCK_SIZE),
            length: 0,
        })
    }

    /// Digest data, updating the internal state.
    pub fn update(
        &mut self,
        mut layouter: impl Layouter<F>,
        mut data: &[Sha256Chip::BlockWord],
    ) -> Result<(), Error> {
        self.length += data.len() * 32;

        // Fill the current block, if possible.
        let remaining = BLOCK_SIZE - self.cur_block.len();
        let (l, r) = data.split_at(min(remaining, data.len()));
        self.cur_block.extend_from_slice(l);
        data = r;

        // If we still don't have a full block, we are done.
        if self.cur_block.len() < BLOCK_SIZE {
            return Ok(());
        }

        // Process the now-full current block.
        self.state = self.chip.compress(
            &mut layouter,
            &self.state,
            self.cur_block[..]
                .try_into()
                .expect("cur_block.len() == BLOCK_SIZE"),
        )?;
        self.cur_block.clear();

        // Process any additional full blocks.
        let mut chunks_iter = data.chunks_exact(BLOCK_SIZE);
        for chunk in &mut chunks_iter {
            self.state = self.chip.initialization(&mut layouter, &self.state)?;
            self.state = self.chip.compress(
                &mut layouter,
                &self.state,
                chunk.try_into().expect("chunk.len() == BLOCK_SIZE"),
            )?;
        }

        // Cache the remaining partial block, if any.
        let rem = chunks_iter.remainder();
        self.cur_block.extend_from_slice(rem);

        Ok(())
    }

    /// Retrieve result and consume hasher instance.
    pub fn finalize(
        mut self,
        mut layouter: impl Layouter<F>,
    ) -> Result<Sha256Digest<Sha256Chip::BlockWord>, Error> {
        // Pad the remaining block
        if !self.cur_block.is_empty() {
            let padding = vec![Sha256Chip::BlockWord::default(); BLOCK_SIZE - self.cur_block.len()];
            self.cur_block.extend_from_slice(&padding);
            self.state = self.chip.initialization(&mut layouter, &self.state)?;
            self.state = self.chip.compress(
                &mut layouter,
                &self.state,
                self.cur_block[..]
                    .try_into()
                    .expect("cur_block.len() == BLOCK_SIZE"),
            )?;
        }
        self.chip
            .digest(&mut layouter, &self.state)
            .map(Sha256Digest)
    }

    /// Convenience function to compute hash of the data. It will handle hasher creation,
    /// data feeding and finalization.
    pub fn digest(
        chip: Sha256Chip,
        mut layouter: impl Layouter<F>,
        data: &[Sha256Chip::BlockWord],
    ) -> Result<Sha256Digest<Sha256Chip::BlockWord>, Error> {
        let mut hasher = Self::new(chip, layouter.namespace(|| "init"))?;
        hasher.update(layouter.namespace(|| "update"), data)?;
        hasher.finalize(layouter.namespace(|| "finalize"))
    }
}

#[derive(Clone, Debug)]
pub struct Sha256Config {
    lookup: SpreadTableConfig,
    message_schedule: MessageScheduleConfig,
    compression: CompressionConfig,
    // Columns where padding is assigned (`a_3, ..., a_8`); these columns are equality constrained
    // and are not used for table inputs.
    padding: [Column<Advice>; 6],
}

#[derive(Debug)]
pub struct Sha256Chip {
    config: Sha256Config,
}

/*
impl Chip<pallas::Base> for Sha256Chip {
    type Config = Sha256Config;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}
*/

impl Sha256Chip {
    pub fn construct(config: Sha256Config) -> Self {
        Sha256Chip { config }
    }

    pub fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        digest: [Column<Advice>; 2],
        extra: [Column<Advice>; 5],
    ) -> Sha256Config {
        // Rename these here for ease of matching the gates to the specification.
        let a_0 = meta.advice_column();
        let a_1 = meta.advice_column();
        let a_2 = meta.advice_column();
        let [a_5, a_8] = digest;
        let [a_3, a_4, a_6, a_7, a_9] = extra;

        // Add advice columns to permutation
        for column in [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8].iter() {
            meta.enable_equality(*column);
        }

        let input_tag = a_0;
        let input_dense = a_1;
        let input_spread = a_2;
        let message_schedule = a_5;
        let padding = [a_3, a_4, a_5, a_6, a_7, a_8];

        let lookup = SpreadTableChip::configure(meta, input_tag, input_dense, input_spread);

        let compression = CompressionConfig::configure(
            meta,
            lookup.input.clone(),
            [a_3, a_4, a_5, a_6, a_7, a_8, a_9],
        );

        let message_schedule = MessageScheduleConfig::configure(
            meta,
            lookup.input.clone(),
            message_schedule,
            [a_3, a_4, a_6, a_7, a_8, a_9],
        );

        Sha256Config {
            lookup,
            message_schedule,
            compression,
            padding,
        }
    }

    /// Loads the lookup table required by this chip into the circuit.
    pub fn load(
        config: Sha256Config,
        layouter: &mut impl Layouter<pallas::Base>,
    ) -> Result<(), Error> {
        SpreadTableChip::load(config.lookup, layouter)
    }

    /// Assign the padding for an assigned, but not yet padded, preimage.
    fn assign_padding(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        preimage: &[AssignedBits<32>],
    ) -> Result<Vec<AssignedBits<32>>, Error> {
        layouter.assign_region(
            || "padding",
            |mut region| {
                let words_utilized = preimage.len();
                let bits_utilized = words_utilized * 32;

                // Padding requires that there are at least 3 unutilized words in the last preimage block.
                // the first padding word appends a `1` bit onto the end of the preimage; the second and
                // third padding words append the number of preimage bits. If there is less than 3 words
                // remaining in the last block, add a full block of padding.
                let mut pad_words = BLOCK_SIZE - (words_utilized % BLOCK_SIZE);
                if pad_words < 3 {
                    pad_words += BLOCK_SIZE;
                }

                let mut padding_iter = iter::once(1u32 << 31)
                    .chain(iter::repeat(0).take(pad_words - 3))
                    .chain(iter::once((bits_utilized >> 32) as u32))
                    .chain(iter::once((bits_utilized & 0xffffffff) as u32));

                let mut padding =
                    Vec::<AssignedBits<32>>::with_capacity(words_utilized + pad_words);
                let mut i = 0;
                let mut row = 0;
                loop {
                    for col in self.config.padding.iter() {
                        if let Some(word) = padding_iter.next() {
                            padding.push(AssignedBits::<32>::assign(
                                &mut region,
                                || format!("pad word {}", i),
                                *col,
                                row,
                                Some(word),
                            )?);
                        } else {
                            return Ok(padding);
                        }
                        i += 1;
                    }
                    row += 1;
                }
            },
        )
    }

    /// Assign the preimage and its padding.
    fn assign_preimage_and_padding(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        preimage: &[BlockWord],
    ) -> Result<Vec<AssignedBits<32>>, Error> {
        layouter.assign_region(
            || "preimage and padding",
            |mut region| {
                let words_utilized = preimage.len();
                let bits_utilized = words_utilized * 32;

                // Padding requires that there are at least 3 unutilized words in the last preimage block.
                // the first padding word appends a `1` bit onto the end of the preimage; the second and
                // third padding words append the number of preimage bits. If there is less than 3 words
                // remaining in the last block, add a full block of padding.
                let mut pad_words = BLOCK_SIZE - (words_utilized % BLOCK_SIZE);
                if pad_words < 3 {
                    pad_words += BLOCK_SIZE;
                }

                let mut preimage_iter = preimage.iter();
                let mut padding_iter = iter::once(1u32 << 31)
                    .chain(iter::repeat(0).take(pad_words - 3))
                    .chain(iter::once((bits_utilized >> 32) as u32))
                    .chain(iter::once((bits_utilized & 0xffffffff) as u32));

                let mut padded =
                    Vec::<AssignedBits<32>>::with_capacity(words_utilized + pad_words);
                let mut i = 0;
                let mut row = 0;
                loop {
                    for col in self.config.padding.iter() {
                        if let Some(word) = preimage_iter.next() {
                            padded.push(AssignedBits::<32>::assign(
                                &mut region,
                                || format!("preimage word {}", i),
                                *col,
                                row,
                                word.0,
                            )?);
                        } else if let Some(word) = padding_iter.next() {
                            padded.push(AssignedBits::<32>::assign(
                                &mut region,
                                || format!("preimage word {} (pad word {})", i, i - words_utilized),
                                *col,
                                row,
                                Some(word),
                            )?);
                        } else {
                            return Ok(padded);
                        }
                        i += 1;
                    }
                    row += 1;
                }
            },
        )
    }

    fn initialization_vector(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
    ) -> Result<State, Error> {
        self.config.compression.initialize_with_iv(layouter, IV)
    }

    fn initialization(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        init_state: State,
    ) -> Result<State, Error> {
        self.config
            .compression
            .initialize_with_state(layouter, init_state)
    }

    fn compress(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        initialized_state: State,
        input: [AssignedBits<32>; BLOCK_SIZE],
    ) -> Result<State, Error> {
        let (_, w_halves) = self.config.message_schedule.process_assigned(layouter, input)?;
        self.config.compression.compress(layouter, initialized_state, w_halves)
    }

    fn assign_digest(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        state: State,
    ) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
        self.config.compression.digest(layouter, state)
    }

    /// Hash, without adding padding, an assigned preimage of 32-bit words. The preimage must be
    /// assigned in columns which are equality constrained.
    pub fn hash_nopad(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        preimage: &[AssignedBits<32>],
    ) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
        assert_eq!(
            preimage.len() % BLOCK_SIZE,
            0,
            "preimage length must be divisible by block size",
        );

        let mut blocks = preimage.chunks(BLOCK_SIZE);

        // Process the first block.
        let mut state = self.initialization_vector(&mut layouter)?;
        state = self.compress(
            &mut layouter,
            state.clone(),
            blocks.next().unwrap().to_vec().try_into().unwrap(),
        )?;

        // Process any additional blocks.
        for block in blocks {
            state = self.initialization(&mut layouter, state.clone())?;
            state = self.compress(
                &mut layouter,
                state.clone(),
                block.to_vec().try_into().unwrap(),
            )?;
        }

        // Assign the digest.
        self.assign_digest(&mut layouter, state)
    }

    /// Pad and hash an assigned (and unpadded) preimage of 32-bit words. The preimage must be
    /// assigned in columns which are equality constrained.
    pub fn hash(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        preimage: &[AssignedBits<32>],
    ) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
        let pad = self.assign_padding(&mut layouter, preimage)?;
        let padded: Vec<AssignedBits<32>> = preimage.iter()
            .chain(pad.iter())
            .cloned()
            .collect();
        self.hash_nopad(layouter, &padded)
    }

    /// Pad and hash an unassigned (and unpadded) preimage of 32-bit words.
    pub fn hash_unassigned(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        preimage: &[BlockWord],
    ) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
        let padded = self.assign_preimage_and_padding(&mut layouter, &preimage)?;
        self.hash_nopad(layouter, &padded)
    }

    /// Hash, without adding padding, an unassigned preimage of 32-bit words.
    pub fn hash_unassigned_nopad(
        &self,
        mut layouter: impl Layouter<pallas::Base>,
        preimage: &[BlockWord],
    ) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
        // Assign the preimage.
        let preimage = layouter.assign_region(
            || "preimage",
            |mut region| {
                let len = preimage.len();
                let mut preimage_iter = preimage.iter();
                let mut preimage = Vec::<AssignedBits<32>>::with_capacity(len);
                let mut i = 0;
                let mut row = 0;
                loop {
                    for col in self.config.padding.iter() {
                        if let Some(word) = preimage_iter.next() {
                            preimage.push(AssignedBits::<32>::assign(
                                &mut region,
                                || format!("preimage word {}", i),
                                *col,
                                row,
                                word.0,
                            )?);
                            i += 1;
                        } else {
                            return Ok(preimage);
                        }
                    }
                    row += 1;
                }
            },
        )?;

        self.hash_nopad(layouter, &preimage)
    }

    // TODO: bit packing
    // pub fn hash_bits() {}
}

pub fn sha256_hash(
    layouter: impl Layouter<pallas::Base>,
    config: Sha256Config,
    preimage: &[AssignedBits<32>],
) -> Result<[AssignedBits<32>; DIGEST_SIZE], Error> {
    Sha256Chip::construct(config).hash(layouter, preimage)
}

#[test]
fn test_sha256_chip() {
    use halo2_proofs::{
        circuit::SimpleFloorPlanner,
        dev::MockProver,
        plonk::{Circuit, Instance},
    };

    use self::table16::msg_schedule_test_input;

    // `N` is the number of words in the preimage. 
    struct MyCircuit<const N: usize> {
        preimage: Vec<BlockWord>,
    }

    #[derive(Clone, Debug)]
    struct MyConfig {
        sha: Sha256Config,
        digest: [Column<Advice>; 2],
        digest_pi: [Column<Instance>; 2],
    }

    /*
    struct PrivateInputs {
        preimage: Vec<BlockWord>,
        num_words: usize,
    }

    impl PrivateInputs {
        fn with_witness(preimage: &[u8]) -> Self {
            let num_bytes = preimage.len();
            let num_words = num_bytes / 4;
            assert_eq!(num_bytes % 4, 0, "preimage length is not divisible by word length (4 bytes)");
            let preimage: Vec<BlockWord> = preimage
                .chunks(4)
                .map(|bytes| BlockWord(Some(u32::from_le_bytes(bytes.try_into().unwrap()))))
                .collect();
            PrivateInputs {
                preimage,
                num_words,
            }
        }
    }
    */

    impl<const N: usize> Circuit<pallas::Base> for MyCircuit<N> {
        type Config = MyConfig;
        type FloorPlanner = SimpleFloorPlanner;

        fn without_witnesses(&self) -> Self {
            MyCircuit {
                preimage: vec![BlockWord(None); N],
            }
        }

        fn configure(meta: &mut ConstraintSystem<pallas::Base>) -> Self::Config {
            let digest = [meta.advice_column(), meta.advice_column()];
            let extra = [
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
                meta.advice_column(),
            ];
            let sha = Sha256Chip::configure(meta, digest, extra);

            let digest_pi = [meta.instance_column(), meta.instance_column()];
            for pi_col in digest_pi.iter() {
                meta.enable_equality(*pi_col);
            }

            MyConfig {
                sha,
                digest,
                digest_pi,
            }
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<pallas::Base>,
        ) -> Result<(), Error> {
            let chip = Sha256Chip::construct(config.sha.clone());
            Sha256Chip::load(config.sha.clone(), &mut layouter)?;

            let digest = chip.hash_unassigned_nopad(layouter.namespace(|| "sha256"), &self.preimage)?;
            /*
            assert_eq!(
                &digest.iter().map(|assigned| assigned.value_u32().unwrap()).collect::<Vec<u32>>(),
                &[
                    885875301, 3540547218, 1155417372, 115889104, 3102768991, 411527067, 1627810339,
                    1767489117,
                ],
            );
            */

            // Constrain equality between the 8 digest cells and the 8 public inputs.
            let mut digest_iter = digest.iter();
            for row in 0..4 {
                for pi_col in config.digest_pi.iter() {
                    layouter.constrain_instance(
                        digest_iter.next().unwrap().cell(),
                        *pi_col,
                        row,
                    )?;
                }
            }

            Ok(())
        }
    }

    impl<const N: usize> MyCircuit<N> {
        fn generate_public_inputs(digest: [u32; DIGEST_SIZE]) -> Vec<Vec<pallas::Base>> {
            let mut pub_inputs = vec![Vec::with_capacity(4), Vec::with_capacity(4)];
            let mut digest_iter = digest.iter().map(|word| *word as u64);
            for _row in 0..4 {
                pub_inputs[0].push(pallas::Base::from(digest_iter.next().unwrap()));
                pub_inputs[1].push(pallas::Base::from(digest_iter.next().unwrap()));
            }
            pub_inputs
        }
    }

    // Create a message of 31 blocks.
    const PREIMAGE_BLOCKS: usize = 31;
    const PREIMAGE_WORDS: usize = PREIMAGE_BLOCKS * BLOCK_SIZE;

    let mut preimage = Vec::with_capacity(PREIMAGE_WORDS);
    let abc_block = msg_schedule_test_input();
    for _ in 0..31 {
        preimage.extend_from_slice(&abc_block);
    }

    let circ = MyCircuit::<PREIMAGE_WORDS> {
        preimage,
    };

    /*
    let expected_digest = [
        BlockWord(Some(885875301)),
        BlockWord(Some(3540547218)),
        BlockWord(Some(1155417372)),
        BlockWord(Some(115889104)),
        BlockWord(Some(3102768991)),
        BlockWord(Some(411527067)),
        BlockWord(Some(1627810339)),
        BlockWord(Some(1767489117)),
    ];
    */
    let expected_digest = [
        885875301, 3540547218, 1155417372, 115889104, 3102768991, 411527067, 1627810339, 1767489117,
    ];
    let pub_inputs = MyCircuit::<PREIMAGE_WORDS>::generate_public_inputs(expected_digest);

    let prover = MockProver::run(17, &circ, pub_inputs).unwrap();
    assert!(prover.verify().is_ok());
}
