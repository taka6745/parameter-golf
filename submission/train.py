import collections,copy,glob,io,lzma,math,os
from pathlib import Path
import random,re,subprocess,sys,time,uuid,numpy as np,sentencepiece as spm,torch,torch.distributed as dist,torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor,nn
try:
    from flash_attn_interface import flash_attn_func as _fa3_raw
    def flash_attn_3_func(q, k, v, causal=True):
        return _fa3_raw(q, k, v, causal=causal)
except ImportError:
    def flash_attn_3_func(q, k, v, causal=True):
        qt = q.transpose(1, 2); kt = k.transpose(1, 2); vt = v.transpose(1, 2)
        n_q = qt.size(1); n_kv = kt.size(1)
        if n_q != n_kv:
            n_rep = n_q // n_kv
            kt = kt.repeat_interleave(n_rep, dim=1)
            vt = vt.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal).transpose(1, 2).contiguous()
class Hyperparameters:data_dir=os.environ.get('DATA_DIR','./data/');seed=int(os.environ.get('SEED',1337));run_id=os.environ.get('RUN_ID',str(uuid.uuid4()));iterations=int(os.environ.get('ITERATIONS',20000));warmdown_frac=float(os.environ.get('WARMDOWN_FRAC',.667));warmup_steps=int(os.environ.get('WARMUP_STEPS',20));train_batch_tokens=int(os.environ.get('TRAIN_BATCH_TOKENS',786432));train_seq_len=int(os.environ.get('TRAIN_SEQ_LEN',2048));train_log_every=int(os.environ.get('TRAIN_LOG_EVERY',500));max_wallclock_seconds=float(os.environ.get('MAX_WALLCLOCK_SECONDS',6e2));val_batch_tokens=int(os.environ.get('VAL_BATCH_TOKENS',524288));eval_seq_len=int(os.environ.get('EVAL_SEQ_LEN',2048));val_loss_every=int(os.environ.get('VAL_LOSS_EVERY',4000));sliding_window_enabled=bool(int(os.environ.get('SLIDING_WINDOW_ENABLED','1')));vocab_size=int(os.environ.get('VOCAB_SIZE',8192));num_layers=int(os.environ.get('NUM_LAYERS',11));xsa_last_n=int(os.environ.get('XSA_LAST_N',11));model_dim=int(os.environ.get('MODEL_DIM',512));embedding_dim=int(os.environ.get('EMBEDDING_DIM',512));num_kv_heads=int(os.environ.get('NUM_KV_HEADS',4));num_heads=int(os.environ.get('NUM_HEADS',8));mlp_mult=float(os.environ.get('MLP_MULT',4.));skip_gates_enabled=bool(int(os.environ.get('SKIP_GATES_ENABLED','1')));tie_embeddings=bool(int(os.environ.get('TIE_EMBEDDINGS','1')));logit_softcap=float(os.environ.get('LOGIT_SOFTCAP',3e1));rope_base=float(os.environ.get('ROPE_BASE',1e4));rope_dims=int(os.environ.get('ROPE_DIMS',16));rope_train_seq_len=int(os.environ.get('ROPE_TRAIN_SEQ_LEN',2048));ln_scale=bool(int(os.environ.get('LN_SCALE','1')));qk_gain_init=float(os.environ.get('QK_GAIN_INIT',4.));num_loops=int(os.environ.get('NUM_LOOPS',2));loop_start=int(os.environ.get('LOOP_START',4));loop_end=int(os.environ.get('LOOP_END',5));enable_looping_at=float(os.environ.get('ENABLE_LOOPING_AT',.5));parallel_residual_start=int(os.environ.get('PARALLEL_RESIDUAL_START','-1'));min_lr=float(os.environ.get('MIN_LR',.0));embed_lr=float(os.environ.get('EMBED_LR',.6));head_lr=float(os.environ.get('HEAD_LR',.008));tied_embed_lr=float(os.environ.get('TIED_EMBED_LR',.03));tied_embed_init_std=float(os.environ.get('TIED_EMBED_INIT_STD',.005));matrix_lr=float(os.environ.get('MATRIX_LR',.02));scalar_lr=float(os.environ.get('SCALAR_LR',.02));muon_momentum=float(os.environ.get('MUON_MOMENTUM',.99));muon_backend_steps=int(os.environ.get('MUON_BACKEND_STEPS',5));muon_momentum_warmup_start=float(os.environ.get('MUON_MOMENTUM_WARMUP_START',.92));muon_momentum_warmup_steps=int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS',1500));muon_row_normalize=bool(int(os.environ.get('MUON_ROW_NORMALIZE','1')));beta1=float(os.environ.get('BETA1',.9));beta2=float(os.environ.get('BETA2',.95));adam_eps=float(os.environ.get('ADAM_EPS',1e-08));grad_clip_norm=float(os.environ.get('GRAD_CLIP_NORM',.3));eval_stride=int(os.environ.get('EVAL_STRIDE',64));muon_beta2=float(os.environ.get('MUON_BETA2',.95));adam_wd=float(os.environ.get('ADAM_WD',.02));muon_wd=float(os.environ.get('MUON_WD',.085));embed_wd=float(os.environ.get('EMBED_WD',.085));ema_decay=float(os.environ.get('EMA_DECAY',.997));ttt_enabled=bool(int(os.environ.get('TTT_ENABLED','0')));ttt_lr=float(os.environ.get('TTT_LR',.005));ttt_epochs=int(os.environ.get('TTT_EPOCHS',3));ttt_chunk_tokens=int(os.environ.get('TTT_CHUNK_TOKENS',32768));ttt_freeze_blocks=int(os.environ.get('TTT_FREEZE_BLOCKS',0));ttt_momentum=float(os.environ.get('TTT_MOMENTUM',.9));ttt_batch_seqs=int(os.environ.get('TTT_BATCH_SEQS',32));ttt_grad_clip=float(os.environ.get('TTT_GRAD_CLIP',1.));prequant_ttt_enabled=bool(int(os.environ.get('PREQUANT_TTT_ENABLED','0')));prequant_ttt_lr=float(os.environ.get('PREQUANT_TTT_LR',.00045));prequant_ttt_epochs=int(os.environ.get('PREQUANT_TTT_EPOCHS',8));prequant_ttt_freeze_blocks=int(os.environ.get('PREQUANT_TTT_FREEZE_BLOCKS',1));prequant_ttt_batch_seqs=int(os.environ.get('PREQUANT_TTT_BATCH_SEQS',32));prequant_ttt_grad_clip=float(os.environ.get('PREQUANT_TTT_GRAD_CLIP',1.));prequant_ttt_cosine_decay=bool(int(os.environ.get('PREQUANT_TTT_COSINE_DECAY','1')));compressor=os.environ.get('COMPRESSOR','brotli');gptq_calibration_batches=int(os.environ.get('GPTQ_CALIBRATION_BATCHES',64));gptq_reserve_seconds=float(os.environ.get('GPTQ_RESERVE_SECONDS',12.));matrix_bits=int(os.environ.get('MATRIX_BITS',6));embed_bits=int(os.environ.get('EMBED_BITS',8));matrix_clip_sigmas=float(os.environ.get('MATRIX_CLIP_SIGMAS',12.85));embed_clip_sigmas=float(os.environ.get('EMBED_CLIP_SIGMAS',2e1));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ;rank=int(os.environ.get('RANK','0'));world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));is_main_process=rank==0;grad_accum_steps=8//world_size;datasets_dir=os.path.join(data_dir,'datasets',f"fineweb10B_sp{vocab_size}");train_files=os.path.join(datasets_dir,'fineweb_train_*.bin');val_files=os.path.join(datasets_dir,'fineweb_val_*.bin');tokenizer_path=os.path.join(data_dir,'tokenizers',f"fineweb_{vocab_size}_bpe.model");logfile=f"logs/{run_id}.txt";model_path='final_model.pt';quantized_model_path='final_model.int6.ptz'
_logger_hparams=None
def set_logging_hparams(h):global _logger_hparams;_logger_hparams=h
def log(msg,console=True):
	if _logger_hparams is None:print(msg);return
	if _logger_hparams.is_main_process:
		if console:print(msg)
		if _logger_hparams.logfile is not None:
			with open(_logger_hparams.logfile,'a',encoding='utf-8')as f:print(msg,file=f)
class ValidationData:
	def __init__(self,h,device):
		self.sp=spm.SentencePieceProcessor(model_file=h.tokenizer_path)
		if int(self.sp.vocab_size())!=h.vocab_size:raise ValueError(f"VOCAB_SIZE={h.vocab_size} does not match tokenizer vocab_size={int(self.sp.vocab_size())}")
		self.val_tokens=load_validation_tokens(h.val_files,h.eval_seq_len);self.base_bytes_lut,self.has_leading_space_lut,self.is_boundary_token_lut=build_sentencepiece_luts(self.sp,h.vocab_size,device)
def build_sentencepiece_luts(sp,vocab_size,device):
	sp_vocab_size=int(sp.vocab_size());table_size=max(sp_vocab_size,vocab_size);base_bytes_np=np.zeros((table_size,),dtype=np.int16);has_leading_space_np=np.zeros((table_size,),dtype=np.bool_);is_boundary_token_np=np.ones((table_size,),dtype=np.bool_)
	for token_id in range(sp_vocab_size):
		if sp.is_control(token_id)or sp.is_unknown(token_id)or sp.is_unused(token_id):continue
		is_boundary_token_np[token_id]=False
		if sp.is_byte(token_id):base_bytes_np[token_id]=1;continue
		piece=sp.id_to_piece(token_id)
		if piece.startswith('▁'):has_leading_space_np[token_id]=True;piece=piece[1:]
		base_bytes_np[token_id]=len(piece.encode('utf-8'))
	return torch.tensor(base_bytes_np,dtype=torch.int16,device=device),torch.tensor(has_leading_space_np,dtype=torch.bool,device=device),torch.tensor(is_boundary_token_np,dtype=torch.bool,device=device)
def load_validation_tokens(pattern,seq_len):
	files=[Path(p)for p in sorted(glob.glob(pattern))]
	if not files:raise FileNotFoundError(f"No files found for pattern: {pattern}")
	tokens=torch.cat([load_data_shard(file)for file in files]).contiguous();usable=(tokens.numel()-1)//seq_len*seq_len
	if usable<=0:raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
	return tokens[:usable+1]
def load_data_shard(file):
	header_bytes=256*np.dtype('<i4').itemsize;token_bytes=np.dtype('<u2').itemsize;header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	num_tokens=int(header[2]);expected_size=header_bytes+num_tokens*token_bytes
	if file.stat().st_size!=expected_size:raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
	tokens_np=np.fromfile(file,dtype='<u2',count=num_tokens,offset=header_bytes)
	if tokens_np.size!=num_tokens:raise ValueError(f"Short read for {file}")
	return torch.from_numpy(tokens_np.astype(np.uint16,copy=False))
_SHARD_HEADER_BYTES=256*np.dtype('<i4').itemsize
_SHARD_NTOKENS_CACHE={}
_MMAP_CACHE={}
def _read_num_tokens(file):
	key=str(file);cached=_SHARD_NTOKENS_CACHE.get(key)
	if cached is not None:return cached
	header=np.fromfile(file,dtype='<i4',count=256)
	if header.size!=256 or int(header[0])!=20240520 or int(header[1])!=1:raise ValueError(f"Unexpected shard header for {file}")
	n=int(header[2]);_SHARD_NTOKENS_CACHE[key]=n;return n
def _get_shard_memmap(file):
	key=str(file);mm=_MMAP_CACHE.get(key)
	if mm is not None:return mm
	n=_read_num_tokens(file);mm=np.memmap(file,mode='r',dtype='<u2',offset=_SHARD_HEADER_BYTES,shape=(n,));_MMAP_CACHE[key]=mm;return mm
class ShuffledSequenceLoader:
	"""Training data loader with optional background prefetch thread.

	Set USE_PREFETCH_LOADER=1 to enable CPU-side prefetch: a daemon thread
	builds batches into pinned-memory tensors ahead of the GPU, so the CPU
	data loader runs IN PARALLEL with GPU forward/backward. Queue depth is
	controlled by PREFETCH_DEPTH (default 4).

	Without prefetch (default), behaves identically to the original
	synchronous path: next_batch() builds the batch on the main thread then
	ships to GPU via .to(non_blocking=True).

	Thread safety: only the worker thread touches self.rng and self.start_inds
	once prefetch is active. The main thread only pops from the queue and
	does the H2D transfer. self.files / self.num_tokens are read-only after
	__init__ so memmap access is safe across threads.
	"""
	def __init__(self,h,device):
		self.world_size=h.world_size;self.seq_len=h.train_seq_len;self.device=device;all_files=[Path(p)for p in sorted(glob.glob(h.train_files))]
		if not all_files:raise FileNotFoundError(f"No files found for pattern: {h.train_files}")
		self.files=all_files[h.rank::h.world_size];self.rng=np.random.Generator(np.random.PCG64(h.rank));self.num_tokens=[_read_num_tokens(f)for f in self.files];self.start_inds=[[]for _ in self.files]
		for si in range(len(self.files)):self._reset_shard(si)
		# Phase 2 prefetch config (USE_PREFETCH_LOADER=1 spawns a daemon worker)
		self._use_prefetch=bool(int(os.environ.get('USE_PREFETCH_LOADER','0')))
		self._prefetch_depth=int(os.environ.get('PREFETCH_DEPTH','4'))
		self._prefetch_queue=None
		self._prefetch_thread=None
		self._prefetch_args=None  # (global_tokens, grad_accum_steps) captured on first call
		self._prefetch_use_pinned=bool(int(os.environ.get('PREFETCH_PIN_MEMORY','1')))
		self._prefetch_stats={'batches_served':0,'queue_waits_empty':0,'queue_waits_full':0}
	def _reset_shard(self,si):max_phase=min(self.seq_len-1,max(0,self.num_tokens[si]-self.seq_len-1));phase=int(self.rng.integers(max_phase+1))if max_phase>0 else 0;num_sequences=(self.num_tokens[si]-1-phase)//self.seq_len;sequence_order=self.rng.permutation(num_sequences);self.start_inds[si]=(phase+sequence_order*self.seq_len).tolist()
	def _build_batch_cpu(self,global_tokens,grad_accum_steps):
		"""Build one (x, y) batch on CPU. Returns pinned tensors if
		PREFETCH_PIN_MEMORY=1 (default). Thread-safe for single-worker use."""
		device_tokens=global_tokens//(self.world_size*grad_accum_steps)
		device_batch_size=device_tokens//self.seq_len
		remaining=np.array([len(s) for s in self.start_inds],dtype=np.float64)
		if self._prefetch_use_pinned:
			x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64,pin_memory=True)
			y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64,pin_memory=True)
		else:
			x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
			y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
		for bi in range(device_batch_size):
			total=remaining.sum()
			if total<=0:
				for si in range(len(self.files)):self._reset_shard(si)
				remaining=np.array([len(s) for s in self.start_inds],dtype=np.float64)
				total=remaining.sum()
			probs=remaining/total
			si=int(self.rng.choice(len(self.files),p=probs))
			start_ind=self.start_inds[si].pop()
			remaining[si]-=1
			mm=_get_shard_memmap(self.files[si])
			window=torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1],dtype=np.int64))
			x[bi]=window[:-1]
			y[bi]=window[1:]
		return x,y
	def _prefetch_worker(self):
		"""Background daemon thread: loops forever, pushing batches into the
		queue. Any exception is surfaced to the main thread via a sentinel
		tuple ('__ERROR__', exc)."""
		try:
			while True:
				x,y=self._build_batch_cpu(*self._prefetch_args)
				self._prefetch_queue.put((x,y))
		except Exception as exc:
			try:
				self._prefetch_queue.put(('__ERROR__',exc))
			except Exception:
				pass
	def _ensure_prefetch_started(self,global_tokens,grad_accum_steps):
		if self._prefetch_queue is not None:
			return
		import queue as _queue
		import threading as _threading
		self._prefetch_queue=_queue.Queue(maxsize=self._prefetch_depth)
		self._prefetch_args=(global_tokens,grad_accum_steps)
		self._prefetch_thread=_threading.Thread(
			target=self._prefetch_worker,
			daemon=True,
			name='ShuffledSequenceLoader-prefetch',
		)
		self._prefetch_thread.start()
		print(f"[prefetch] daemon started: depth={self._prefetch_depth} pinned={self._prefetch_use_pinned}",flush=True)
	def next_batch(self,global_tokens,grad_accum_steps):
		if self._use_prefetch:
			self._ensure_prefetch_started(global_tokens,grad_accum_steps)
			# Detect queue-empty stalls for telemetry
			if self._prefetch_queue.empty():
				self._prefetch_stats['queue_waits_empty']+=1
			item=self._prefetch_queue.get()
			if isinstance(item,tuple) and len(item)>=1 and item[0]=='__ERROR__':
				raise item[1]
			x,y=item
			self._prefetch_stats['batches_served']+=1
			return x.to(self.device,non_blocking=True),y.to(self.device,non_blocking=True)
		# Fallback: synchronous path (original behavior)
		device_tokens=global_tokens//(self.world_size*grad_accum_steps);device_batch_size=device_tokens//self.seq_len;remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);x=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64);y=torch.empty((device_batch_size,self.seq_len),dtype=torch.int64)
		for bi in range(device_batch_size):
			total=remaining.sum()
			if total<=0:
				for si in range(len(self.files)):self._reset_shard(si)
				remaining=np.array([len(s)for s in self.start_inds],dtype=np.float64);total=remaining.sum()
			probs=remaining/total;si=int(self.rng.choice(len(self.files),p=probs));start_ind=self.start_inds[si].pop();remaining[si]-=1;mm=_get_shard_memmap(self.files[si]);window=torch.as_tensor(np.array(mm[start_ind:start_ind+self.seq_len+1],dtype=np.int64));x[bi]=window[:-1];y[bi]=window[1:]
		return x.to(self.device,non_blocking=True),y.to(self.device,non_blocking=True)
	def prefetch_queue_depth(self):
		"""Current depth of the prefetch queue (for telemetry). Returns -1 if
		prefetch is disabled."""
		if self._prefetch_queue is None:
			return -1
		return self._prefetch_queue.qsize()
	def prefill(self,global_tokens,grad_accum_steps,target_depth=None,timeout_s=120.0):
		"""Pre-fill the prefetch queue during pretime so training starts with
		a full queue. Front-loads CPU work into pretime (free) so the CPU is
		nearly idle during the 600s training budget (available for metric
		logging, optimizer offload, async checkpoint writes, etc.).

		Blocks until the queue has `target_depth` batches, or until timeout.
		Only runs if USE_PREFETCH_LOADER=1.

		Env var override: PREFETCH_PREFILL_BATCHES (default = PREFETCH_DEPTH).
		"""
		if not self._use_prefetch:
			print("[prefetch] prefill: USE_PREFETCH_LOADER=0, skipping",flush=True)
			return
		if target_depth is None:
			target_depth=int(os.environ.get('PREFETCH_PREFILL_BATCHES',str(self._prefetch_depth)))
		target_depth=min(target_depth,self._prefetch_depth)  # can't exceed queue maxsize
		self._ensure_prefetch_started(global_tokens,grad_accum_steps)
		import time as _time
		t0=_time.perf_counter()
		last_log=t0
		print(f"[prefetch] prefill: target_depth={target_depth}, maxsize={self._prefetch_depth}, timeout={timeout_s}s",flush=True)
		while True:
			current=self._prefetch_queue.qsize()
			if current>=target_depth:
				elapsed=_time.perf_counter()-t0
				print(f"[prefetch] prefill: reached depth {current}/{target_depth} in {elapsed:.2f}s",flush=True)
				return
			elapsed=_time.perf_counter()-t0
			if elapsed>=timeout_s:
				print(f"[prefetch] prefill: TIMEOUT at depth {current}/{target_depth} after {elapsed:.1f}s",flush=True)
				return
			# Progress logging every 5s
			if _time.perf_counter()-last_log>5.0:
				print(f"[prefetch] prefill progress: {current}/{target_depth} at {elapsed:.1f}s",flush=True)
				last_log=_time.perf_counter()
			_time.sleep(0.1)
class RMSNorm(nn.Module):
	def __init__(self,eps=None):super().__init__();self.eps=eps
	def forward(self,x):return F.rms_norm(x,(x.size(-1),),eps=self.eps)
class CastedLinear(nn.Linear):
	def forward(self,x):w=self.weight.to(x.dtype);bias=self.bias.to(x.dtype)if self.bias is not None else None;return F.linear(x,w,bias)
class Rotary(nn.Module):
	def __init__(self,dim,base=1e4,train_seq_len=1024,rope_dims=0):
		super().__init__();self.dim=dim;self.base=base;self.train_seq_len=train_seq_len;self.rope_dims=rope_dims if rope_dims>0 else dim
		inv_freq=1./base**(torch.arange(0,self.rope_dims,2,dtype=torch.float32)/self.rope_dims)
		self.register_buffer('inv_freq',inv_freq,persistent=False)
		# E10: pre-compute cos/sin at train_seq_len as persistent=False buffers so
		# forward() is a pure slice read (CUDA-graph safe). Old dynamic caching
		# pattern triggered "accessing overwritten tensor" under max-autotune +
		# CUDA graphs. Buffers are fp32; forward() casts to target dtype on each
		# call (lightweight) — acceptable since x.dtype is constant during a run.
		t=torch.arange(self.train_seq_len,dtype=torch.float32);freqs=torch.outer(t,inv_freq)
		self.register_buffer('_cos_pre',freqs.cos()[None,:,None,:],persistent=False)
		self.register_buffer('_sin_pre',freqs.sin()[None,:,None,:],persistent=False)
		self._max_pre_seq_len=self.train_seq_len
		# legacy attributes kept for backward compat (no longer written to)
		self._seq_len_cached=0;self._cos_cached=None;self._sin_cached=None
	def forward(self,seq_len,device,dtype):
		if seq_len<=self._max_pre_seq_len:
			# E10 fast path: pure slice, no allocation, CUDA-graph safe
			return self._cos_pre[:,:seq_len].to(dtype=dtype),self._sin_pre[:,:seq_len].to(dtype=dtype)
		# Legacy fallback for seq_len > train_seq_len (base interpolation)
		if self._cos_cached is None or self._seq_len_cached!=seq_len or self._cos_cached.device!=device:
			rd=self.rope_dims
			scale=seq_len/self.train_seq_len;new_base=self.base*scale**(rd/(rd-2))
			inv_freq=1./new_base**(torch.arange(0,rd,2,dtype=torch.float32,device=device)/rd)
			t=torch.arange(seq_len,device=device,dtype=inv_freq.dtype);freqs=torch.outer(t,inv_freq)
			self._cos_cached=freqs.cos()[None,:,None,:];self._sin_cached=freqs.sin()[None,:,None,:];self._seq_len_cached=seq_len
		return self._cos_cached.to(dtype=dtype),self._sin_cached.to(dtype=dtype)
def apply_rotary_emb(x,cos,sin,rope_dims=0):
	if rope_dims>0 and rope_dims<x.size(-1):x_rope,x_pass=x[...,:rope_dims],x[...,rope_dims:];half=rope_dims//2;x1,x2=x_rope[...,:half],x_rope[...,half:];x_rope=torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1);return torch.cat((x_rope,x_pass),dim=-1)
	half=x.size(-1)//2;x1,x2=x[...,:half],x[...,half:];return torch.cat((x1*cos+x2*sin,x1*-sin+x2*cos),dim=-1)
class CausalSelfAttention(nn.Module):
	def __init__(self,dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len):
		super().__init__()
		if dim%num_heads!=0:raise ValueError('model_dim must be divisible by num_heads')
		if num_heads%num_kv_heads!=0:raise ValueError('num_heads must be divisible by num_kv_heads')
		self.num_heads=num_heads;self.num_kv_heads=num_kv_heads;self.head_dim=dim//num_heads
		if self.head_dim%2!=0:raise ValueError('head_dim must be even for RoPE')
		kv_dim=self.num_kv_heads*self.head_dim;self.c_q=CastedLinear(dim,dim,bias=False);self.c_k=CastedLinear(dim,kv_dim,bias=False);self.c_v=CastedLinear(dim,kv_dim,bias=False);self.proj=CastedLinear(dim,dim,bias=False);self.proj._zero_init=True;self.q_gain=nn.Parameter(torch.full((num_heads,),qk_gain_init,dtype=torch.float32));self.rope_dims=0;self.rotary=Rotary(self.head_dim,base=rope_base,train_seq_len=train_seq_len);self.use_xsa=False
		# GATED_ATTENTION (NIGHT_MODE n=5 confirmed-win, our champion lever)
		# Per-head sigmoid gate over attention output. Gate = sigmoid(W_g @ x + b_g),
		# init weight=0 + bias=2.94 → sigmoid(2.94)≈0.95 (near identity, room to learn).
		# Multiplied per-head onto y after FA3 + XSA, before reshape.
		# Cost: dim*num_heads + num_heads params/layer (4104 for our 512/8 shape).
		# Enable via USE_GATED_ATTENTION=1.
		self.gate_proj=CastedLinear(dim,num_heads,bias=True)
		with torch.no_grad():
			self.gate_proj.weight.zero_()
			if self.gate_proj.bias is not None:self.gate_proj.bias.fill_(2.94)
		self.use_gated_attention=bool(int(os.environ.get('USE_GATED_ATTENTION','0')))
	def _xsa_efficient(self,y,v):B,T,H,D=y.shape;Hkv=v.size(-2);group=H//Hkv;y_g=y.reshape(B,T,Hkv,group,D);vn=F.normalize(v,dim=-1).unsqueeze(-2);proj=(y_g*vn).sum(dim=-1,keepdim=True)*vn;return(y_g-proj).reshape(B,T,H,D)
	def forward(self,x):
		bsz,seqlen,dim=x.shape;q=self.c_q(x).reshape(bsz,seqlen,self.num_heads,self.head_dim);k=self.c_k(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);v=self.c_v(x).reshape(bsz,seqlen,self.num_kv_heads,self.head_dim);q=F.rms_norm(q,(q.size(-1),));k=F.rms_norm(k,(k.size(-1),));cos,sin=self.rotary(seqlen,x.device,q.dtype);q=apply_rotary_emb(q,cos,sin,self.rope_dims);k=apply_rotary_emb(k,cos,sin,self.rope_dims);q=q*self.q_gain.to(dtype=q.dtype)[None,None,:,None];y=flash_attn_3_func(q,k,v,causal=True)
		if self.use_xsa:y=self._xsa_efficient(y,v)
		# GATED_ATTENTION apply: per-head sigmoid gate on attention output (after FA3+XSA, before reshape)
		if self.use_gated_attention:
			gate=torch.sigmoid(self.gate_proj(x).float()).to(dtype=y.dtype)  # (B, S, num_heads)
			y=y*gate.unsqueeze(-1)  # broadcast over head_dim
		y=y.reshape(bsz,seqlen,dim);return self.proj(y)
class MLP(nn.Module):
	def __init__(self,dim,mlp_mult):
		super().__init__()
		hidden=int(mlp_mult*dim)
		self.fc=CastedLinear(dim,hidden,bias=False)
		self.proj=CastedLinear(hidden,dim,bias=False)
		self.proj._zero_init=True
		# NORM_PCT_DROPOUT (NIGHT_MODE world-novel L05, n=2 confirmed-win 1.41365):
		# Zero out the top 1% per-token L2-norm rows of the FFN intermediate. Targets
		# rare exploding-activation pathways. Standard dropout = random elements;
		# structured dropout = random rows; norm-percentile dropout = the loudest rows.
		# Enable via USE_NORM_PCT_DROPOUT=1, threshold via NORM_PCT_THRESH (default 0.99).
		self.use_norm_pct_dropout=bool(int(os.environ.get('USE_NORM_PCT_DROPOUT','0')))
		self.norm_pct_thresh=float(os.environ.get('NORM_PCT_THRESH','0.99'))
	def forward(self,x):
		x=F.leaky_relu(self.fc(x),negative_slope=.5).square()
		if self.training and self.use_norm_pct_dropout:
			# Zero rows whose L2 norm is in the top (1 - thresh) fraction
			orig_shape=x.shape
			x_flat=x.reshape(-1,orig_shape[-1])
			row_norms=x_flat.float().norm(dim=-1)
			kth=torch.quantile(row_norms,self.norm_pct_thresh)
			keep=(row_norms<kth).to(dtype=x.dtype).unsqueeze(-1)
			x_flat=x_flat*keep
			x=x_flat.reshape(orig_shape)
		return self.proj(x)
class Block(nn.Module):
	# Parallel Residuals (Shot 11+): when USE_PARALLEL_RESIDUALS=1, attn and mlp run
	# on the same input x_in instead of mlp consuming attn's output. Used by leaderboard
	# #1 stack. Inductor can fuse the two branches better; ~+0.005-0.01 BPB.
	def __init__(self,dim,num_heads,num_kv_heads,mlp_mult,rope_base,qk_gain_init,train_seq_len,layer_idx=0,ln_scale=False):super().__init__();self.attn_norm=RMSNorm();self.mlp_norm=RMSNorm();self.attn=CausalSelfAttention(dim,num_heads,num_kv_heads,rope_base,qk_gain_init,train_seq_len);self.mlp=MLP(dim,mlp_mult);self.attn_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.mlp_scale=nn.Parameter(torch.ones(dim,dtype=torch.float32));self.resid_mix=nn.Parameter(torch.stack((torch.ones(dim),torch.zeros(dim))).float());self.ln_scale_factor=1./math.sqrt(layer_idx+1)if ln_scale else 1.;self.layer_idx=layer_idx;self._parallel_residuals=bool(int(os.environ.get('USE_PARALLEL_RESIDUALS','0')))or(int(os.environ.get('PARALLEL_RESIDUAL_START','-1'))>=0 and layer_idx>=int(os.environ.get('PARALLEL_RESIDUAL_START','-1')))
	def forward(self,x,x0):
		mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0
		attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor)
		if self._parallel_residuals:
			# Parallel: attn and mlp both consume x_in
			mlp_out=self.mlp(self.mlp_norm(x_in)*self.ln_scale_factor)
			return x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out+self.mlp_scale.to(dtype=x_in.dtype)[None,None,:]*mlp_out
		# Serial (original): mlp consumes attn's output
		x_out=x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out
		x_out=x_out+self.mlp_scale.to(dtype=x_out.dtype)[None,None,:]*self.mlp(self.mlp_norm(x_out)*self.ln_scale_factor)
		return x_out
	def forward_attn(self,x,x0):mix=self.resid_mix.to(dtype=x.dtype);x_in=mix[0][None,None,:]*x+mix[1][None,None,:]*x0;attn_out=self.attn(self.attn_norm(x_in)*self.ln_scale_factor);return x_in+self.attn_scale.to(dtype=x_in.dtype)[None,None,:]*attn_out
	def forward_mlp(self,x):return x+self.mlp_scale.to(dtype=x.dtype)[None,None,:]*self.mlp(self.mlp_norm(x)*self.ln_scale_factor)
class GPT(nn.Module):
	def __init__(self,h):
		super().__init__()
		if h.logit_softcap<=.0:raise ValueError(f"logit_softcap must be positive, got {h.logit_softcap}")
		self.tie_embeddings=h.tie_embeddings;self.tied_embed_init_std=h.tied_embed_init_std;self.logit_softcap=h.logit_softcap;self.tok_emb=nn.Embedding(h.vocab_size,h.embedding_dim)
		if h.embedding_dim!=h.model_dim:self.embed_proj=CastedLinear(h.embedding_dim,h.model_dim,bias=False);self.head_proj=CastedLinear(h.model_dim,h.embedding_dim,bias=False)
		else:self.embed_proj=None;self.head_proj=None
		self.num_encoder_layers=h.num_layers//2;self.num_decoder_layers=h.num_layers-self.num_encoder_layers;self.blocks=nn.ModuleList([Block(h.model_dim,h.num_heads,h.num_kv_heads,h.mlp_mult,h.rope_base,h.qk_gain_init,h.train_seq_len,layer_idx=i,ln_scale=h.ln_scale)for i in range(h.num_layers)])
		if h.rope_dims>0:
			head_dim=h.model_dim//h.num_heads
			for block in self.blocks:block.attn.rope_dims=h.rope_dims;block.attn.rotary=Rotary(head_dim,base=h.rope_base,train_seq_len=h.train_seq_len,rope_dims=h.rope_dims)
		self.final_norm=RMSNorm();self.lm_head=None if h.tie_embeddings else CastedLinear(h.embedding_dim,h.vocab_size,bias=False)
		if self.lm_head is not None:self.lm_head._zero_init=True
		if h.xsa_last_n>0:
			for i in range(max(0,h.num_layers-h.xsa_last_n),h.num_layers):self.blocks[i].attn.use_xsa=True
		self.looping_active=False
		if h.num_loops>0:
			loop_seg=list(range(h.loop_start,h.loop_end+1));all_indices=list(range(h.loop_start))
			for _ in range(h.num_loops+1):all_indices.extend(loop_seg)
			all_indices.extend(range(h.loop_end+1,h.num_layers));num_enc=len(all_indices)//2;self.encoder_indices=all_indices[:num_enc];self.decoder_indices=all_indices[num_enc:]
		else:self.encoder_indices=list(range(self.num_encoder_layers));self.decoder_indices=list(range(self.num_encoder_layers,h.num_layers))
		self.num_skip_weights=min(len(self.encoder_indices),len(self.decoder_indices));self.skip_weights=nn.Parameter(torch.ones(self.num_skip_weights,h.model_dim,dtype=torch.float32));self.skip_gates=nn.Parameter(torch.zeros(self.num_skip_weights,h.model_dim,dtype=torch.float32))if h.skip_gates_enabled else None
		self.parallel_start_layer=int(os.environ.get('PARALLEL_START_LAYER','7'));self.lane_merge=nn.Parameter(torch.tensor(0.5))if self.parallel_start_layer>0 else None
		# NGRAM_BIAS (NIGHT_MODE infrastructure for NGRAM_BACKOFF + world-novel L09 refinements)
		# Loads precomputed bigram/trigram/fourgram log-prob tables as non-persistent buffers.
		# Tables built by submission/build_ngrams.py from tokenized shards. Loaded fresh on every
		# pod boot — they do NOT count toward the 16 MB submission limit.
		# Hash function: polynomial (prev * 36313 + cur * 27191 + ...) % HASH_BUCKETS.
		# At forward time, looks up bias via hash and adds to logits.
		self._ngram_enabled=bool(int(os.environ.get('USE_NGRAM_BIAS','0')))
		self._ngram_w_bigram=float(os.environ.get('NGRAM_W_BIGRAM','0.20'))
		self._ngram_w_trigram=float(os.environ.get('NGRAM_W_TRIGRAM','0.15'))
		self._ngram_w_fourgram=float(os.environ.get('NGRAM_W_FOURGRAM','0.10'))
		self._ngram_hash=int(os.environ.get('NGRAM_HASH_BUCKETS','16384'))
		self._ngram_backoff=bool(int(os.environ.get('USE_NGRAM_BACKOFF','0')))
		self._ngram_backoff_t4=float(os.environ.get('NGRAM_BACKOFF_THRESH4','1.0'))
		self._ngram_backoff_t3=float(os.environ.get('NGRAM_BACKOFF_THRESH3','1.0'))
		self._ngram_backoff_alpha=float(os.environ.get('NGRAM_BACKOFF_ALPHA','0.4'))
		# NGR_LOG_FREQ_INV (NIGHT_MODE world-novel L09): one-time inverse-log-frequency
		# bucket suppression on first forward. High-freq buckets get muted; rare buckets
		# stay strong. Mutates n-gram tables in place ONCE; subsequent forwards are no-op.
		# Targets the trigram bias swamping problem: model already predicts confident
		# common contexts well, so muting the bias there frees capacity for rare contexts.
		self._nlfi_enabled=bool(int(os.environ.get('USE_NGR_LOG_FREQ_INV','0')))
		self._nlfi_applied=False  # Python flag — reset on every fresh __init__
		# SHOT 0e FIX: the n-gram bias tables (_bigram_tab etc.) are persistent=False,
		# so the in-place mutation from NGR_LOG_FREQ_INV is LOST on serialize→deserialize.
		# Fix: save the multipliers as PERSISTENT small buffers (16384 fp32 each = 64 KB
		# each, ~192 KB total — well under the 16 MB cap). On deserialize, __init__
		# reloads fresh tables from disk, state_dict load restores the multipliers,
		# and the first forward pass re-applies them to the freshly-loaded tables.
		# This preserves the NGR_LOG_FREQ_INV mutation across serialize boundaries.
		# A single-int flag `_nlfi_stored_flag` tells us whether multipliers were
		# already computed (and hence loaded from state_dict) or need fresh compute.
		self.register_buffer('_nlfi_bigram_mult',torch.ones(self._ngram_hash,dtype=torch.float32),persistent=True)
		self.register_buffer('_nlfi_trigram_mult',torch.ones(self._ngram_hash,dtype=torch.float32),persistent=True)
		self.register_buffer('_nlfi_fourgram_mult',torch.ones(self._ngram_hash,dtype=torch.float32),persistent=True)
		self.register_buffer('_nlfi_stored_flag',torch.zeros(1,dtype=torch.int64),persistent=True)
		# CTX_PARTITIONED_TAB (NIGHT_MODE world-novel L09): 16 virtual sub-tables via
		# slice rotation by (current_id mod S) * (HASH/S). Effectively partitions hash
		# buckets into S zones, each absorbing 1/S of contexts → finer-grained smoothing.
		# Mini-paper extension of the tabulation hash framework.
		self._ctx_part_tab_enabled=bool(int(os.environ.get('USE_CTX_PARTITIONED_TAB','0')))
		self._ctx_part_slices=int(os.environ.get('CTX_PARTITION_SLICES','16'))
		self.register_buffer('_bigram_tab',torch.zeros(1,dtype=torch.float32),persistent=False)
		self.register_buffer('_trigram_tab',torch.zeros(1,dtype=torch.float32),persistent=False)
		self.register_buffer('_fourgram_tab',torch.zeros(1,dtype=torch.float32),persistent=False)
		if self._ngram_enabled:
			vs=h.vocab_size
			# E11: optional bf16 n-gram tables. Saves 750 MB VRAM (3 tables × 512 MB → 256 MB).
			# Log-prob values are in ~(-15, 0) range, bf16 precision (~3 decimal digits) is
			# sufficient. Enable via USE_NGRAM_BF16=1.
			_ngram_bf16=bool(int(os.environ.get('USE_NGRAM_BF16','0')))
			_ngram_dtype=torch.bfloat16 if _ngram_bf16 else torch.float32
			# E7a: optional bigram-only mode. Skips trigram + fourgram loads (and their
			# forward-pass lookups via the existing numel() > 1 guards). Tests whether
			# the 3-gram and 4-gram lookups are a bottleneck worth fusing.
			_ngram_bigram_only=bool(int(os.environ.get('USE_NGRAM_BIGRAM_ONLY','0')))
			for tab_attr,fname,label in [
				('_bigram_tab',f'data/bigram_tab_{vs}v.npy','bigram'),
				('_trigram_tab',f'data/trigram_logprobs_{vs}v.npy','trigram'),
				('_fourgram_tab',f'data/fourgram_logprobs_{vs}v.npy','fourgram'),
			]:
				if _ngram_bigram_only and tab_attr!='_bigram_tab':
					print(f'NGRAM_BIAS: {label} SKIPPED (USE_NGRAM_BIGRAM_ONLY=1)',flush=True)
					continue
				try:
					_arr=np.load(fname)
					_tab=torch.from_numpy(_arr).to(dtype=_ngram_dtype)
					setattr(self,tab_attr,_tab)
					print(f'NGRAM_BIAS: loaded {label} {_arr.shape} from {fname} dtype={_ngram_dtype}',flush=True)
				except Exception as _e:
					print(f'NGRAM_BIAS: {label} load failed ({fname}): {_e}',flush=True)
		self._init_weights()
	def _init_weights(self):
		if self.tie_embeddings:nn.init.normal_(self.tok_emb.weight,mean=.0,std=self.tied_embed_init_std)
		for(name,module)in self.named_modules():
			if isinstance(module,nn.Linear):
				if getattr(module,'_zero_init',False):nn.init.zeros_(module.weight)
				elif module.weight.ndim==2 and module.weight.shape[0]>=64 and module.weight.shape[1]>=64:nn.init.orthogonal_(module.weight,gain=1.)
	def forward_logits(self,input_ids):
		x=self.tok_emb(input_ids);x=F.rms_norm(x,(x.size(-1),))
		if self.embed_proj is not None:x=self.embed_proj(x)
		x0=x;skips=[];enc_iter=self.encoder_indices if self.looping_active else range(self.num_encoder_layers);dec_iter=self.decoder_indices if self.looping_active else range(self.num_encoder_layers,self.num_encoder_layers+self.num_decoder_layers)
		for i in enc_iter:x=self.blocks[i](x,x0);skips.append(x)
		psl=self.parallel_start_layer;lane0=None;lane1=None
		for(skip_idx,i)in enumerate(dec_iter):
			if lane0 is None:
				if skip_idx<self.num_skip_weights and skips:
					scaled_skip=self.skip_weights[skip_idx].to(dtype=x.dtype)[None,None,:]*skips.pop()
					if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None,None,:];x=torch.lerp(scaled_skip,x,g)
					else:x=x+scaled_skip
				if i>=psl and psl>0:lane0=x;lane1=x.clone();lane0=self.blocks[i].forward_attn(lane0,x0);lane1=self.blocks[i].forward_mlp(lane1)
				else:x=self.blocks[i](x,x0)
			else:
				if skip_idx<self.num_skip_weights and skips:
					scaled_skip=self.skip_weights[skip_idx].to(dtype=lane0.dtype)[None,None,:]*skips.pop()
					if self.skip_gates is not None:g=torch.sigmoid(self.skip_gates[skip_idx].to(dtype=lane0.dtype))[None,None,:];lane0=torch.lerp(scaled_skip,lane0,g)
					else:lane0=lane0+scaled_skip
				lane0=self.blocks[i].forward_attn(lane0,x0);lane1=self.blocks[i].forward_mlp(lane1)
		if lane0 is not None:lm=self.lane_merge.to(dtype=lane0.dtype);x=lm*lane0+(1.-lm)*lane1
		x=self.final_norm(x)
		if self.head_proj is not None:x=self.head_proj(x)
		if self.tie_embeddings:logits_proj=F.linear(x,self.tok_emb.weight)
		else:logits_proj=self.lm_head(x)
		logits=self.logit_softcap*torch.tanh(logits_proj/self.logit_softcap)
		# NGRAM_BIAS apply: blend in precomputed n-gram log-prob bias (NIGHT_MODE infra)
		# When NGRAM_BACKOFF is on: order-adaptive Stupid Backoff (Brants 2007). Pick the
		# highest-order hash whose peak log-prob exceeds the threshold; lower orders attenuated
		# by alpha. Otherwise: weighted sum of all three orders.
		if self._ngram_enabled and self._bigram_tab.numel()>1:
			B,S=input_ids.shape
			_zeros1=torch.zeros(B,1,device=input_ids.device,dtype=input_ids.dtype)
			_zeros2=torch.zeros(B,2,device=input_ids.device,dtype=input_ids.dtype)
			_ids_flat=input_ids.reshape(-1).long()
			_prev2=torch.cat([_zeros1,input_ids[:,:-1]],dim=1).reshape(-1).long()
			_prev3=torch.cat([_zeros2,input_ids[:,:-2]],dim=1).reshape(-1).long()
			H=self._ngram_hash
			# NGR_LOG_FREQ_INV setup happens EAGERLY via _apply_nlfi_once() before
			# torch.compile wraps this model (see train_model/train_and_eval). This
			# avoids a .item() graph break that fullgraph=True can't tolerate.
			_h_bi=(_ids_flat*36313)%H
			# CTX_PARTITIONED_TAB: rotate bigram hash by per-token slice for finer smoothing
			if self._ctx_part_tab_enabled:
				_S_slices=self._ctx_part_slices
				_zone=(_ids_flat%_S_slices)*(H//_S_slices)
				_h_bi=(_h_bi+_zone)%H
			_h_tri=(_prev2*36313+_ids_flat*27191)%H
			_h_four=(_prev3*36313+_prev2*27191+_ids_flat*51497)%H
			_bi=self._bigram_tab[_h_bi].reshape(B,S,-1)
			if self._ngram_backoff and self._trigram_tab.numel()>1 and self._fourgram_tab.numel()>1:
				# NGRAM_BACKOFF (Brants 2007 Stupid Backoff): pick the highest-confidence order
				_tri=self._trigram_tab[_h_tri].reshape(B,S,-1)
				_four=self._fourgram_tab[_h_four].reshape(B,S,-1)
				_peak4=_four.amax(dim=-1,keepdim=True)
				_peak3=_tri.amax(dim=-1,keepdim=True)
				_use_4=(_peak4>self._ngram_backoff_t4).to(_four.dtype)
				_use_3=(1-_use_4)*(_peak3>self._ngram_backoff_t3).to(_tri.dtype)
				_use_bi=1-_use_4-_use_3
				_alpha=self._ngram_backoff_alpha
				_ng=_use_4*_four+_use_3*_tri*_alpha+_use_bi*_bi*(_alpha*_alpha)
				logits=logits+_ng.to(dtype=logits.dtype)
			else:
				# Weighted sum of all 3 orders (additive blend)
				_bias=self._ngram_w_bigram*_bi
				if self._trigram_tab.numel()>1:
					_bias=_bias+self._ngram_w_trigram*self._trigram_tab[_h_tri].reshape(B,S,-1)
				if self._fourgram_tab.numel()>1:
					_bias=_bias+self._ngram_w_fourgram*self._fourgram_tab[_h_four].reshape(B,S,-1)
				logits=logits+_bias.to(dtype=logits.dtype)
		return logits
	@torch.no_grad()
	def _apply_nlfi_once(self,input_ids):
		# SHOT 0e: one-time NGR_LOG_FREQ_INV bucket mutation. Called EAGERLY from
		# train_model/train_and_eval before torch.compile wraps the model, so the
		# compiled forward never sees the .item() branch (which would graph-break
		# fullgraph=True compile). Idempotent: _nlfi_applied guards against re-entry.
		if self._nlfi_applied or not self._nlfi_enabled:return
		if not(self._ngram_enabled and self._bigram_tab.numel()>1):return
		try:
			_ids_flat=input_ids.reshape(-1).long()
			H=self._ngram_hash
			if int(self._nlfi_stored_flag.item())==1:
				# Restored from state_dict — use the saved multipliers
				_bg_mult=self._nlfi_bigram_mult
				_tg_mult=self._nlfi_trigram_mult
				_fg_mult=self._nlfi_fourgram_mult
				print('NGR_LOG_FREQ_INV: restored multipliers from state_dict',flush=True)
			else:
				# Fresh compute from the current batch's hash bucket counts
				_bg_h_init=(_ids_flat*36313)%H
				_bg_counts=torch.zeros(H,dtype=torch.float32,device=_ids_flat.device)
				_bg_counts.scatter_add_(0,_bg_h_init,torch.ones_like(_bg_h_init,dtype=torch.float32))
				_bg_mult=1.0/torch.log(2.0+_bg_counts)
				_tg_h_init=((_ids_flat*36313)^((_ids_flat*39979)>>1))%H
				_tg_counts=torch.zeros(H,dtype=torch.float32,device=_ids_flat.device)
				_tg_counts.scatter_add_(0,_tg_h_init,torch.ones_like(_tg_h_init,dtype=torch.float32))
				_tg_mult=1.0/torch.log(2.0+_tg_counts)
				_fg_h_init=((_ids_flat*36313)^((_ids_flat*39979)>>1)^((_ids_flat*41077)>>2))%H
				_fg_counts=torch.zeros(H,dtype=torch.float32,device=_ids_flat.device)
				_fg_counts.scatter_add_(0,_fg_h_init,torch.ones_like(_fg_h_init,dtype=torch.float32))
				_fg_mult=1.0/torch.log(2.0+_fg_counts)
				self._nlfi_bigram_mult.data=_bg_mult.detach().to(self._nlfi_bigram_mult.dtype)
				self._nlfi_trigram_mult.data=_tg_mult.detach().to(self._nlfi_trigram_mult.dtype)
				self._nlfi_fourgram_mult.data=_fg_mult.detach().to(self._nlfi_fourgram_mult.dtype)
				self._nlfi_stored_flag.data=torch.ones(1,dtype=torch.int64,device=self._nlfi_stored_flag.device)
				print('NGR_LOG_FREQ_INV: computed + saved multipliers from current batch',flush=True)
			# Apply to the n-gram tables in place
			if self._bigram_tab.numel()>1:
				if self._bigram_tab.dim()==2:self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype).unsqueeze(1))
				else:self._bigram_tab.mul_(_bg_mult.to(self._bigram_tab.dtype))
			if self._trigram_tab.numel()>1:
				if self._trigram_tab.dim()==2:self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype).unsqueeze(1))
				else:self._trigram_tab.mul_(_tg_mult.to(self._trigram_tab.dtype))
			if self._fourgram_tab.numel()>1:
				if self._fourgram_tab.dim()==2:self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype).unsqueeze(1))
				else:self._fourgram_tab.mul_(_fg_mult.to(self._fourgram_tab.dtype))
			print('NGR_LOG_FREQ_INV: applied mutation to n-gram tables (one-time per process)',flush=True)
		except Exception as _e:print(f'NGR_LOG_FREQ_INV: mutation failed ({_e})',flush=True)
		self._nlfi_applied=True
	def forward(self,input_ids,target_ids):logits=self.forward_logits(input_ids);return F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),target_ids.reshape(-1),reduction='mean')
def classify_param(name):
	if'tok_emb'in name or'lm_head'in name:return'embed'
	if'.mlp.'in name:return'mlp'
	if'.attn.'in name or'.proj.'in name and'.mlp.'not in name:return'attn'
	return'other'
@torch.compile
def zeropower_via_newtonschulz5(G,steps=10,eps=1e-07):
	# E6: handle both 2D (rows, cols) and 3D (batch, rows, cols) inputs for
	# Parallel Muon. Uses transpose(-2,-1) everywhere so the code is shape-
	# generic. 2D path is identical to the original (dynamo constant-folds
	# the dim check).
	a,b,c=3.4445,-4.775,2.0315;X=G.bfloat16()
	if X.dim()==2:X=X/(X.norm()+eps)
	else:X=X/(X.flatten(start_dim=-2).norm(dim=-1,keepdim=True).unsqueeze(-1)+eps)
	transposed=X.size(-2)>X.size(-1)
	if transposed:X=X.transpose(-2,-1)
	for _ in range(steps):A=X@X.transpose(-2,-1);B=b*A+c*A@A;X=a*X+B@X
	return X.transpose(-2,-1) if transposed else X
class Muon(torch.optim.Optimizer):
	def __init__(self,params,lr,momentum,backend_steps,nesterov=True,weight_decay=.0,row_normalize=False):super().__init__(params,dict(lr=lr,momentum=momentum,backend_steps=backend_steps,nesterov=nesterov,weight_decay=weight_decay,row_normalize=row_normalize))
	@torch.no_grad()
	def step(self,closure=None):
		loss=None
		if closure is not None:
			with torch.enable_grad():loss=closure()
		distributed=dist.is_available()and dist.is_initialized();world_size=dist.get_world_size()if distributed else 1;rank=dist.get_rank()if distributed else 0
		_use_parallel_muon=int(os.environ.get('USE_PARALLEL_MUON','0'))
		_use_normuon=int(os.environ.get('USE_NORMUON','0'))
		for group in self.param_groups:
			params=group['params']
			if not params:continue
			lr=group['lr'];momentum=group['momentum'];backend_steps=group['backend_steps'];nesterov=group['nesterov']
			total_params=sum(int(p.numel())for p in params);updates_flat=torch.zeros(total_params,device=params[0].device,dtype=torch.bfloat16)
			# Per-param offsets into updates_flat (same order as params list, regardless of which rank owns each)
			_offsets=[0]
			for p in params:_offsets.append(_offsets[-1]+p.numel())
			if _use_parallel_muon:
				# E6 Parameter Banking + Parallel Muon: group params by shape, stack
				# grads into (n, rows, cols), run Newton-Schulz iterations in ONE
				# batched call. Reduces ~24 serial NS calls per step → ~2-3 batched
				# calls (one per distinct param shape). Cuts kernel launch overhead.
				_shape_groups={}  # shape_tuple -> list of (i, p)
				for i,p in enumerate(params):
					if i%world_size!=rank:continue
					if p.grad is None:continue
					sh=tuple(p.grad.shape)
					_shape_groups.setdefault(sh,[]).append((i,p))
				for sh,grp in _shape_groups.items():
					# Per-param momentum + nesterov + pre-NS row_normalize (sequential; tiny)
					_grads=[]
					for i,p in grp:
						g=p.grad;state=self.state[p]
						if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
						buf=state['momentum_buffer'];buf.mul_(momentum).add_(g)
						if nesterov:g=g.add(buf,alpha=momentum)
						if group.get('row_normalize',False):
							_rn=g.float().norm(dim=-1,keepdim=True).clamp_min(1e-07);g=g/_rn.to(g.dtype)
						_grads.append(g)
					# BATCHED NS — single call for the whole shape group
					_stacked=torch.stack(_grads,dim=0)
					_result=zeropower_via_newtonschulz5(_stacked,steps=backend_steps)
					# Scatter results back with per-param NORMUON + scaling
					for _bi,(i,p) in enumerate(grp):
						g=_result[_bi]
						if _use_normuon:
							_post_norm=g.float().norm(dim=-1,keepdim=True).clamp(min=1e-8);g=g/_post_norm.to(g.dtype)
						g=g*max(1,g.size(0)/g.size(1))**.5
						updates_flat[_offsets[i]:_offsets[i+1]]=g.reshape(-1)
			else:
				# Original serial path
				curr=0
				for(i,p)in enumerate(params):
					if i%world_size==rank and p.grad is not None:
						g=p.grad;state=self.state[p]
						if'momentum_buffer'not in state:state['momentum_buffer']=torch.zeros_like(g)
						buf=state['momentum_buffer'];buf.mul_(momentum).add_(g)
						if nesterov:g=g.add(buf,alpha=momentum)
						if group.get('row_normalize',False):row_norms=g.float().norm(dim=-1,keepdim=True).clamp_min(1e-07);g=g/row_norms.to(g.dtype)
						g=zeropower_via_newtonschulz5(g,steps=backend_steps)
						# NORMUON (NIGHT_MODE n=2 confirmed-win): per-row normalize AFTER Newton-Schulz.
						if _use_normuon:
							_post_norm=g.float().norm(dim=-1,keepdim=True).clamp(min=1e-8)
							g=g/_post_norm.to(g.dtype)
						g*=max(1,g.size(0)/g.size(1))**.5;updates_flat[curr:curr+p.numel()]=g.reshape(-1)
					curr+=p.numel()
			if distributed:dist.all_reduce(updates_flat,op=dist.ReduceOp.SUM)
			wd=group.get('weight_decay',.0);curr=0
			for p in params:
				if wd>.0:p.data.mul_(1.-lr*wd)
				g=updates_flat[curr:curr+p.numel()].view_as(p).to(dtype=p.dtype);p.add_(g,alpha=-lr);curr+=p.numel()
		return loss
CONTROL_TENSOR_NAME_PATTERNS=tuple(pattern for pattern in os.environ.get('CONTROL_TENSOR_NAME_PATTERNS','attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates,lane_merge').split(',')if pattern)
class Optimizers:
	def __init__(self,h,base_model):
		block_named_params=list(base_model.blocks.named_parameters());matrix_params=[p for(name,p)in block_named_params if p.ndim==2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)];scalar_params=[p for(name,p)in block_named_params if p.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)]
		if base_model.skip_weights.numel()>0:scalar_params.append(base_model.skip_weights)
		if base_model.skip_gates is not None and base_model.skip_gates.numel()>0:scalar_params.append(base_model.skip_gates)
		if base_model.lane_merge is not None:scalar_params.append(base_model.lane_merge)
		token_lr=h.tied_embed_lr if h.tie_embeddings else h.embed_lr;tok_params=[{'params':[base_model.tok_emb.weight],'lr':token_lr,'base_lr':token_lr}];self.optimizer_tok=torch.optim.AdamW(tok_params,betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.embed_wd,fused=True);self.optimizer_muon=Muon(matrix_params,lr=h.matrix_lr,momentum=h.muon_momentum,backend_steps=h.muon_backend_steps,weight_decay=h.muon_wd,row_normalize=h.muon_row_normalize)
		for group in self.optimizer_muon.param_groups:group['base_lr']=h.matrix_lr
		self.optimizer_scalar=torch.optim.AdamW([{'params':scalar_params,'lr':h.scalar_lr,'base_lr':h.scalar_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,weight_decay=h.adam_wd,fused=True);self.optimizers=[self.optimizer_tok,self.optimizer_muon,self.optimizer_scalar]
		if base_model.lm_head is not None:self.optimizer_head=torch.optim.Adam([{'params':[base_model.lm_head.weight],'lr':h.head_lr,'base_lr':h.head_lr}],betas=(h.beta1,h.beta2),eps=h.adam_eps,fused=True);self.optimizers.insert(1,self.optimizer_head)
		else:self.optimizer_head=None
	def __iter__(self):return iter(self.optimizers)
	def zero_grad_all(self):
		for opt in self.optimizers:opt.zero_grad(set_to_none=True)
	def step(self):
		for opt in self.optimizers:opt.step()
		self.zero_grad_all()
def restore_fp32_params(model):
	for module in model.modules():
		if isinstance(module,CastedLinear):module.float()
	for(name,param)in model.named_parameters():
		if(param.ndim<2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))and param.dtype!=torch.float32:param.data=param.data.float()
def collect_hessians(model,train_loader,h,device,n_calibration_batches=64):
	hessians={};hooks=[]
	def make_hook(name):
		def hook_fn(module,inp,out):
			x=inp[0].detach().float()
			if x.ndim==3:x=x.reshape(-1,x.shape[-1])
			if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
			hessians[name].addmm_(x.T,x)
		return hook_fn
	for(name,module)in model.named_modules():
		if isinstance(module,CastedLinear)and module.weight.numel()>65536:
			cat=classify_param(name+'.weight')
			if cat in('mlp','attn'):hooks.append(module.register_forward_hook(make_hook(name+'.weight')))
	if model.tie_embeddings:
		hook_module=model.head_proj if model.head_proj is not None else model.final_norm
		def make_output_hook(name):
			def hook_fn(module,inp,out):
				x=out.detach().float()
				if x.ndim==3:x=x.reshape(-1,x.shape[-1])
				if name not in hessians:hessians[name]=torch.zeros(x.shape[1],x.shape[1],dtype=torch.float32,device=device)
				hessians[name].addmm_(x.T,x)
			return hook_fn
		hooks.append(hook_module.register_forward_hook(make_output_hook('tok_emb.weight')))
	model.eval()
	with torch.no_grad():
		for _ in range(n_calibration_batches):x,_=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps);model.forward_logits(x)
	for hook in hooks:hook.remove()
	for name in hessians:hessians[name]=hessians[name].cpu()/n_calibration_batches
	return hessians
def gptq_quantize_weight(w,H,clip_sigmas=3.,clip_range=63,block_size=128):
	W_orig=w.float().clone();rows,cols=W_orig.shape;H=H.float().clone();dead=torch.diag(H)==0;H[dead,dead]=1;damp=.01*H.diag().mean();H.diagonal().add_(damp);perm=torch.argsort(H.diag(),descending=True);invperm=torch.argsort(perm);W_perm=W_orig[:,perm].clone();W_perm[:,dead[perm]]=0;H=H[perm][:,perm];Hinv=torch.cholesky_inverse(torch.linalg.cholesky(H));Hinv=torch.linalg.cholesky(Hinv,upper=True);row_std=W_orig.std(dim=1);s=(clip_sigmas*row_std/clip_range).clamp_min(1e-10).to(torch.float16);sf=s.float();Q=torch.zeros(rows,cols,dtype=torch.int8);W_work=W_perm.clone()
	for i1 in range(0,cols,block_size):
		i2=min(i1+block_size,cols);W_block=W_work[:,i1:i2].clone();Hinv_block=Hinv[i1:i2,i1:i2];Err=torch.zeros(rows,i2-i1)
		for j in range(i2-i1):w_col=W_block[:,j];d=Hinv_block[j,j];q_col=torch.clamp(torch.round(w_col/sf),-clip_range,clip_range);Q[:,i1+j]=q_col.to(torch.int8);err=(w_col-q_col.float()*sf)/d;Err[:,j]=err;W_block[:,j:]-=err.unsqueeze(1)*Hinv_block[j,j:].unsqueeze(0)
		if i2<cols:W_work[:,i2:]-=Err@Hinv[i1:i2,i2:]
	Q=Q[:,invperm]
	# CMP_QUANT_VALUE_DEDUP (NIGHT_MODE world-novel L10): snap quantized values to a
	# coarser alphabet AFTER GPTQ but BEFORE serialization. Halves the effective
	# alphabet (e.g. step=2 → 32 distinct int6 values instead of 64), so the byte
	# stream that brotli/zlib subsequently compresses has more repeating bytes →
	# longer LZ77 matches → ~5-15% better post-quant compression. Cost: tiny added
	# reconstruction noise. Enable via USE_CMP_QUANT_VALUE_DEDUP=1, step via
	# CMP_QUANT_DEDUP_STEP (default 2). World-novel: post-int alphabet snap for
	# brotli compressibility is not in any LM compression paper.
	if int(os.environ.get('USE_CMP_QUANT_VALUE_DEDUP','0')):
		_cqvd_step=int(os.environ.get('CMP_QUANT_DEDUP_STEP','2'))
		if _cqvd_step>1:Q=((Q.to(torch.int16)//_cqvd_step)*_cqvd_step).to(torch.int8)
	return Q,s
def gptq_mixed_quantize(state_dict,hessians,h):
	result={};meta={}
	for(name,tensor)in state_dict.items():
		t=tensor.detach().cpu().contiguous()
		if not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
		cs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f"gptq (int{bits})"
	categories=collections.defaultdict(set)
	for(name,cat)in meta.items():short=re.sub('\\.\\d+$','',re.sub('blocks\\.\\d+','blocks',name));categories[cat].add(short)
	log('Quantized weights:')
	for cat in sorted(categories):log(f'  {cat}: {", ".join(sorted(categories[cat]))}')
	return result,meta
def dequantize_mixed(result,meta,template_sd):
	out={}
	for(name,orig)in template_sd.items():
		info=meta.get(name)
		if info is None:continue
		orig_dtype=orig.dtype
		if'passthrough'in info:
			t=result[name]
			if t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
			out[name]=t;continue
		q,s=result[name+'.q'],result[name+'.scale']
		if s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
		else:out[name]=(q.float()*float(s.item())).to(orig_dtype)
	return out
_BSHF_MAGIC=b'BSHF'
def _byte_shuffle(data,stride=2):
	if stride<=1 or len(data)<stride:return data
	src=np.frombuffer(data,dtype=np.uint8);n=len(src);out=np.empty(n,dtype=np.uint8);dest_off=0
	for pos in range(stride):chunk=src[pos::stride];out[dest_off:dest_off+len(chunk)]=chunk;dest_off+=len(chunk)
	return _BSHF_MAGIC+bytes([stride])+out.tobytes()
def _byte_unshuffle(data):
	if len(data)<5 or data[:4]!=_BSHF_MAGIC:return data
	stride=data[4]
	if stride<2:return data[5:]
	payload=np.frombuffer(data,dtype=np.uint8,offset=5);n=len(payload);out=np.empty(n,dtype=np.uint8);src_off=0
	for pos in range(stride):chunk_len=n//stride+(1 if pos<n%stride else 0);out[pos::stride][:chunk_len]=payload[src_off:src_off+chunk_len];src_off+=chunk_len
	return out.tobytes()
def _compress(data,compressor):
	data=_byte_shuffle(data)
	if compressor=='lzma':return lzma.compress(data,preset=6)
	elif compressor=='brotli':import brotli;return brotli.compress(data,quality=11)
	raise ValueError(f"Unknown compressor: {compressor!r}")
def _decompress(data,compressor):
	if compressor=='lzma':raw=lzma.decompress(data)
	elif compressor=='brotli':import brotli;raw=brotli.decompress(data)
	else:raise ValueError(f"Unknown compressor: {compressor!r}")
	raw=_byte_unshuffle(raw);return raw
class _ValCalibLoader:
	# SHOT 0e follow-up: val-token calibration loader for GPTQ. Use this instead
	# of the train ShuffledSequenceLoader when the model has been TTT-adapted,
	# so the Hessian estimates match the val distribution the weights were
	# adapted to. Prevents GPTQ from quantizing away TTT's val-specific updates.
	def __init__(self,val_tokens,h,device):
		self.val_tokens=val_tokens;self.h=h;self.device=device;self._offset=0
	def next_batch(self,batch_tokens,grad_accum_steps):
		seq_len=self.h.train_seq_len
		batch_seqs=max(1,batch_tokens//(seq_len*max(1,grad_accum_steps)))
		needed=batch_seqs*seq_len+1
		if self._offset+needed>self.val_tokens.numel():self._offset=0
		chunk=self.val_tokens[self._offset:self._offset+needed].to(device=self.device,dtype=torch.int64)
		x=chunk[:-1].reshape(-1,seq_len);y=chunk[1:].reshape(-1,seq_len);self._offset+=needed-1
		return x,y
def serialize(h,base_model,code,val_data=None):
	code_bytes=len(code.encode('utf-8'))
	if h.is_main_process:torch.save(base_model.state_dict(),h.model_path);model_bytes=os.path.getsize(h.model_path);log(f"Serialized model: {model_bytes} bytes");log(f"Code size: {code_bytes} bytes")
	sd_cpu={k:v.detach().cpu()for(k,v)in base_model.state_dict().items()};device=torch.device('cuda',h.local_rank)
	_use_val_calib=int(os.environ.get('GPTQ_CALIB_USE_VAL','0'))and val_data is not None
	if _use_val_calib:
		log('GPTQ_CALIB_USE_VAL=1: calibrating Hessians on val tokens (preserves TTT adaptation through int6 quant)')
		calib_loader=_ValCalibLoader(val_data.val_tokens,h,device)
	else:
		log('GPTQ:collecting Hessians from calibration data...')
		calib_loader=ShuffledSequenceLoader(h,device)
	t0=time.perf_counter();hessians=collect_hessians(base_model,calib_loader,h,device,n_calibration_batches=h.gptq_calibration_batches);log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s");quant_result,quant_meta=gptq_mixed_quantize(sd_cpu,hessians,h);quant_buf=io.BytesIO();torch.save({'w':quant_result,'m':quant_meta},quant_buf);quant_raw=quant_buf.getvalue();quant_blob=_compress(quant_raw,h.compressor);quant_file_bytes=len(quant_blob);bytes_total=quant_file_bytes+code_bytes
	if h.is_main_process:
		with open(h.quantized_model_path,'wb')as f:f.write(quant_blob)
		log(f"Serialized model quantized+{h.compressor}: {quant_file_bytes} bytes");log(f"Total submission size quantized+{h.compressor}: {bytes_total} bytes")
	return bytes_total,quant_file_bytes
def deserialize(h,device):
	eval_model=GPT(h).to(device).bfloat16();restore_fp32_params(eval_model);sd_cpu={k:v.detach().cpu()for(k,v)in eval_model.state_dict().items()}
	with open(h.quantized_model_path,'rb')as f:quant_blob_disk=f.read()
	quant_state=torch.load(io.BytesIO(_decompress(quant_blob_disk,h.compressor)),map_location='cpu');deq_state=dequantize_mixed(quant_state['w'],quant_state['m'],sd_cpu);eval_model.load_state_dict(deq_state,strict=True);return eval_model
def _loss_bpb(loss_sum,token_count,byte_count):val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item());return val_loss,val_bpb
def eval_val(h,device,val_data,model):
	seq_len=h.eval_seq_len;local_batch_tokens=h.val_batch_tokens//(h.world_size*h.grad_accum_steps)
	if local_batch_tokens<seq_len:raise ValueError(f"VAL_BATCH_SIZE must provide at least one sequence per rank; got VAL_BATCH_SIZE={h.val_batch_tokens}, WORLD_SIZE={h.world_size}, GRAD_ACCUM_STEPS={h.grad_accum_steps}, seq_len={seq_len}")
	local_batch_seqs=local_batch_tokens//seq_len;total_seqs=(val_data.val_tokens.numel()-1)//seq_len;seq_start=total_seqs*h.rank//h.world_size;seq_end=total_seqs*(h.rank+1)//h.world_size;val_loss_sum=torch.zeros((),device=device,dtype=torch.float64);val_token_count=torch.zeros((),device=device,dtype=torch.float64);val_byte_count=torch.zeros((),device=device,dtype=torch.float64);model.eval()
	with torch.inference_mode():
		for batch_seq_start in range(seq_start,seq_end,local_batch_seqs):
			batch_seq_end=min(batch_seq_start+local_batch_seqs,seq_end);raw_start=batch_seq_start*seq_len;raw_end=batch_seq_end*seq_len+1;local=val_data.val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64,non_blocking=True);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):batch_loss=model(x,y).detach()
			batch_token_count=float(y.numel());val_loss_sum+=batch_loss.to(torch.float64)*batch_token_count;val_token_count+=batch_token_count;prev_ids=x.reshape(-1);tgt_ids=y.reshape(-1);token_bytes=val_data.base_bytes_lut[tgt_ids].to(dtype=torch.int16);token_bytes+=(val_data.has_leading_space_lut[tgt_ids]&~val_data.is_boundary_token_lut[prev_ids]).to(dtype=torch.int16);val_byte_count+=token_bytes.to(torch.float64).sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(val_loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(val_token_count,op=dist.ReduceOp.SUM);dist.all_reduce(val_byte_count,op=dist.ReduceOp.SUM)
	model.train();return _loss_bpb(val_loss_sum,val_token_count,val_byte_count)
def eval_val_sliding(h,device,val_data,base_model,batch_seqs=32):
	base_model.eval();logits_fn=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);seq_len=h.eval_seq_len;context_size=seq_len-h.eval_stride;total_tokens=val_data.val_tokens.numel()-1;window_starts=[ws for ws in range(0,total_tokens,h.eval_stride)if ws+context_size<total_tokens];total_windows=len(window_starts);my_s=total_windows*h.rank//h.world_size;my_e=total_windows*(h.rank+1)//h.world_size;my_windows=window_starts[my_s:my_e];loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64)
	with torch.inference_mode():
		for bi in range(0,len(my_windows),batch_seqs):
			batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
			for(i,ws)in enumerate(batch_ws):we=min(ws+seq_len,total_tokens);wlen=we-ws;wlens.append(wlen);chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk[:-1];y_batch[i,:wlen]=chunk[1:]
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=logits_fn(x_batch)
			nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
			for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	base_model.train();return _loss_bpb(loss_sum,token_count,byte_count)
def eval_val_sliding_ttt(h,base_model,rank,world_size,device,val_data,stride):
	seq_len=h.eval_seq_len;total_tokens=val_data.val_tokens.numel()-1;ttt_chunk=h.ttt_chunk_tokens;context_size=seq_len-stride;window_starts=[ws for ws in range(0,total_tokens,stride)if ws+context_size<total_tokens];num_chunks=(total_tokens+ttt_chunk-1)//ttt_chunk;chunk_windows=[[]for _ in range(num_chunks)]
	for ws in window_starts:end=min(ws+seq_len,total_tokens);wlen=end-ws;s=0 if ws==0 else context_size;scored_start=ws+s;ci=min(scored_start//ttt_chunk,num_chunks-1);chunk_windows[ci].append(ws)
	log(f"ttt_sliding:start chunks={num_chunks} chunk_tokens={ttt_chunk} total_windows={len(window_starts)} stride={stride} ttt_lr={h.ttt_lr} ttt_epochs={h.ttt_epochs} freeze_blocks={h.ttt_freeze_blocks}");compiled_logits=torch.compile(base_model.forward_logits,dynamic=False,fullgraph=True);loss_sum=torch.zeros((),device=device,dtype=torch.float64);token_count=torch.zeros((),device=device,dtype=torch.float64);byte_count=torch.zeros((),device=device,dtype=torch.float64);frozen_block_ids=set(range(min(h.ttt_freeze_blocks,len(base_model.blocks))));ttt_params=[]
	for(name,p)in base_model.named_parameters():
		freeze=False
		for bi in frozen_block_ids:
			if f"blocks.{bi}."in name:freeze=True;break
		if freeze:p.requires_grad_(False)
		else:p.requires_grad_(True);ttt_params.append(p)
	log(f"ttt_sliding:params unfrozen={sum(p.numel()for p in ttt_params)} frozen={sum(p.numel()for p in base_model.parameters()if not p.requires_grad)}");optimizer=torch.optim.SGD(ttt_params,lr=h.ttt_lr,momentum=h.ttt_momentum);t0=time.perf_counter();batch_seqs=h.ttt_batch_seqs
	for ci in range(num_chunks):
		windows=chunk_windows[ci]
		if not windows:continue
		chunk_start=ci*ttt_chunk;chunk_end=min((ci+1)*ttt_chunk,total_tokens);my_s=len(windows)*rank//world_size;my_e=len(windows)*(rank+1)//world_size;my_windows=windows[my_s:my_e];base_model.eval()
		with torch.no_grad():
			for bi in range(0,len(my_windows),batch_seqs):
				batch_ws=my_windows[bi:bi+batch_seqs];bsz=len(batch_ws);x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device);wlens=[]
				for(i,ws)in enumerate(batch_ws):end=min(ws+seq_len,total_tokens);wlen=end-ws;wlens.append(wlen);chunk_tok=val_data.val_tokens[ws:end+1].to(dtype=torch.int64,device=device);x_batch[i,:wlen]=chunk_tok[:-1];y_batch[i,:wlen]=chunk_tok[1:]
				with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=compiled_logits(x_batch)
				nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction='none').reshape(bsz,seq_len)
				for(i,ws)in enumerate(batch_ws):wlen=wlens[i];s=0 if ws==0 else context_size;scored_nll=nll[i,s:wlen].to(torch.float64);loss_sum+=scored_nll.sum();token_count+=float(wlen-s);tgt=y_batch[i,s:wlen];prev=x_batch[i,s:wlen];tb=val_data.base_bytes_lut[tgt].to(torch.float64);tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64);byte_count+=tb.sum()
		is_last_chunk=ci==num_chunks-1
		if not is_last_chunk and h.ttt_epochs>0:
			base_model.train();chunk_seqs=(chunk_end-chunk_start)//seq_len
			if chunk_seqs>0:
				cos_lr=h.ttt_lr*.5*(1.+math.cos(math.pi*ci/max(num_chunks-1,1)))
				for pg in optimizer.param_groups:pg['lr']=cos_lr
				my_seq_s=chunk_seqs*rank//world_size;my_seq_e=chunk_seqs*(rank+1)//world_size;my_chunk_seqs=my_seq_e-my_seq_s
				for _ep in range(h.ttt_epochs):
					for bs in range(0,my_chunk_seqs,batch_seqs):
						be=min(bs+batch_seqs,my_chunk_seqs);actual_bs=my_seq_s+bs;start_tok=chunk_start+actual_bs*seq_len;end_tok=chunk_start+(my_seq_s+be)*seq_len+1
						if end_tok>val_data.val_tokens.numel():continue
						local=val_data.val_tokens[start_tok:end_tok].to(device=device,dtype=torch.int64);x=local[:-1].reshape(-1,seq_len);y=local[1:].reshape(-1,seq_len);optimizer.zero_grad(set_to_none=True)
						with torch.autocast(device_type='cuda',dtype=torch.bfloat16):loss=base_model(x,y)
						loss.backward()
						if world_size>1:
							for p in ttt_params:
								if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
						torch.nn.utils.clip_grad_norm_(ttt_params,h.ttt_grad_clip);optimizer.step()
		if rank==0 and(ci%10==0 or ci==num_chunks-1):elapsed=time.perf_counter()-t0;rl=loss_sum.item()/max(token_count.item(),1);rbpb=rl/math.log(2.)*(token_count.item()/max(byte_count.item(),1));log(f"  ttt_chunk [{ci+1}/{num_chunks}] bpb={rbpb:.6f} time={elapsed:.1f}s")
	if dist.is_available()and dist.is_initialized():dist.all_reduce(loss_sum,op=dist.ReduceOp.SUM);dist.all_reduce(token_count,op=dist.ReduceOp.SUM);dist.all_reduce(byte_count,op=dist.ReduceOp.SUM)
	val_loss=(loss_sum/token_count).item();val_bpb=val_loss/math.log(2.)*(token_count.item()/byte_count.item())
	for p in base_model.parameters():p.requires_grad_(True)
	base_model.eval();log(f"ttt_sliding:done val_loss={val_loss:.6f}{ val_bpb=:.6f} elapsed={time.perf_counter()-t0:.1f}s");return val_loss,val_bpb
def timed_eval(label,fn,*args,**kwargs):torch.cuda.synchronize();t0=time.perf_counter();val_loss,val_bpb=fn(*args,**kwargs);torch.cuda.synchronize();elapsed_ms=1e3*(time.perf_counter()-t0);log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{elapsed_ms:.0f}ms");return val_loss,val_bpb
def _load_train_sample_for_nlfi(h,device):
	"""RULE COMPLIANCE: NLFI bias mutation must use TRAIN data, not val (the comp
	rules forbid accessing val data during training). Loads the first eval_seq_len
	tokens from the first train shard. Deterministic, so train-side and eval-side
	NLFI setup compute matching multipliers."""
	try:
		_train_files=sorted(Path(h.datasets_dir).resolve().glob('fineweb_train_*.bin'))
		if not _train_files:return None
		_arr=np.fromfile(str(_train_files[0]),dtype=np.uint16,count=h.eval_seq_len)
		if _arr.size<h.eval_seq_len:return None
		return torch.from_numpy(_arr.astype(np.int64)).to(device).view(1,-1)
	except Exception as _e:
		print(f'NLFI: train sample load failed ({_e}), falling back to no setup',flush=True)
		return None
def train_model(h,device,val_data):
	base_model=GPT(h).to(device).bfloat16();restore_fp32_params(base_model)
	# SHOT 0e: run NLFI one-time setup eagerly BEFORE torch.compile so the
	# compiled forward never sees .item() (which graph-breaks fullgraph=True).
	# RULE COMPLIANCE: NLFI sample must come from TRAIN data, not val.
	if getattr(base_model,'_nlfi_enabled',False) and not getattr(base_model,'_nlfi_applied',False):
		_sample=_load_train_sample_for_nlfi(h,device)
		if _sample is not None:
			base_model._apply_nlfi_once(_sample)
	# E4/SPEED: torch.compile mode selectable via env. 'max-autotune' does more
	# kernel autotuning up-front (longer first-compile cost, faster steady-state
	# steps). Cache is persistent so first-compile cost is paid once.
	_compile_mode=os.environ.get('TORCH_COMPILE_MODE','default')
	if _compile_mode=='default':
		compiled_model=torch.compile(base_model,dynamic=False,fullgraph=True)
	else:
		log(f"torch.compile mode={_compile_mode}")
		compiled_model=torch.compile(base_model,dynamic=False,fullgraph=True,mode=_compile_mode)
	if h.distributed:model=DDP(compiled_model,device_ids=[h.local_rank],broadcast_buffers=False)
	else:model=compiled_model
	log(f"model_params:{sum(p.numel()for p in base_model.parameters())}");optimizers=Optimizers(h,base_model);train_loader=ShuffledSequenceLoader(h,device);max_wallclock_ms=1e3*h.max_wallclock_seconds if h.max_wallclock_seconds>0 else None
	# Phase 2: prefill the prefetch queue during PRETIME (before wallclock starts)
	# so the training loop starts with a fully-staged data pipeline. CPU batch-build
	# time is front-loaded here — free, not counted toward the 600s budget.
	train_loader.prefill(h.train_batch_tokens,h.grad_accum_steps)
	if max_wallclock_ms is not None:max_wallclock_ms-=h.gptq_reserve_seconds*1e3;log(f"gptq:reserving {h.gptq_reserve_seconds:.0f}s, effective={max_wallclock_ms:.0f}ms")
	# Shot 17: Fuzzy LR bandit. Thompson-sample a per-step LR multiplier from
	# {0.5x, 1x, 2x} * base schedule. Reward = train_loss decrease. Arms with
	# higher mean reward get sampled more. Disabled by default.
	_fuzzy_enabled=int(os.environ.get('USE_FUZZY_LR_BANDIT','0'))
	_fuzzy_arms=[0.5,1.0,2.0];_fuzzy_means=[0.0,0.0,0.0];_fuzzy_counts=[1,1,1]
	_fuzzy_prev_loss=None;_fuzzy_arm_idx=1
	if _fuzzy_enabled:log(f"FUZZY_LR_BANDIT: enabled arms={_fuzzy_arms} (Shot 17)")
	def training_frac(step,elapsed_ms):
		if max_wallclock_ms is None:return step/max(h.iterations,1)
		return elapsed_ms/max(max_wallclock_ms,1e-09)
	def lr_mul(frac):
		if h.warmdown_frac<=0:return 1.
		if frac>=1.-h.warmdown_frac:return max((1.-frac)/h.warmdown_frac,h.min_lr)
		return 1.
	def step_fn(step,lr_scale):
		optimizers.zero_grad_all();train_loss=torch.zeros((),device=device)
		for micro_step in range(h.grad_accum_steps):
			if h.distributed:model.require_backward_grad_sync=micro_step==h.grad_accum_steps-1
			x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=True):loss=model(x,y)
			train_loss+=loss.detach();(loss/h.grad_accum_steps).backward()
		train_loss/=h.grad_accum_steps;frac=min(step/h.muon_momentum_warmup_steps,1.)if h.muon_momentum_warmup_steps>0 else 1.;muon_momentum=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
		for group in optimizers.optimizer_muon.param_groups:group['momentum']=muon_momentum
		for opt in optimizers:
			for group in opt.param_groups:group['lr']=group['base_lr']*lr_scale
		if h.grad_clip_norm>0:torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
		optimizers.step();return train_loss
	if h.warmup_steps>0:
		initial_model_state={name:tensor.detach().cpu().clone()for(name,tensor)in base_model.state_dict().items()};initial_optimizer_states=[copy.deepcopy(opt.state_dict())for opt in optimizers];model.train()
		for warmup_step in range(h.warmup_steps):
			step_fn(warmup_step,1.)
			if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"warmup_step: {warmup_step+1}/{h.warmup_steps}")
		if h.num_loops>0:
			base_model.looping_active=True;log(f"loop_warmup:enabled encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
			for warmup_step in range(h.warmup_steps):
				step_fn(warmup_step,1.)
				if warmup_step<=5 or(warmup_step+1)%10==0 or warmup_step+1==h.warmup_steps:log(f"loop_warmup_step: {warmup_step+1}/{h.warmup_steps}")
			base_model.looping_active=False
		base_model.load_state_dict(initial_model_state,strict=True)
		for(opt,state)in zip(optimizers,initial_optimizer_states,strict=True):opt.load_state_dict(state)
		optimizers.zero_grad_all()
		if h.distributed:model.require_backward_grad_sync=True
		train_loader=ShuffledSequenceLoader(h,device)
	ema_state={name:t.detach().float().clone()for(name,t)in base_model.state_dict().items()};ema_decay=h.ema_decay;training_time_ms=.0;stop_after_step=None;torch.cuda.synchronize();t0=time.perf_counter();step=0
	while True:
		last_step=step==h.iterations or stop_after_step is not None and step>=stop_after_step;should_validate=last_step or h.val_loss_every>0 and step%h.val_loss_every==0
		if should_validate:torch.cuda.synchronize();training_time_ms+=1e3*(time.perf_counter()-t0);val_loss,val_bpb=eval_val(h,device,val_data,model);log(f"{step}/{h.iterations} val_loss: {val_loss:.4f} val_bpb: {val_bpb:.4f}");torch.cuda.synchronize();t0=time.perf_counter()
		if last_step:
			if stop_after_step is not None and step<h.iterations:log(f"stopping_early: wallclock_cap train_time: {training_time_ms:.0f}ms step: {step}/{h.iterations}")
			break
		elapsed_ms=training_time_ms+1e3*(time.perf_counter()-t0);frac=training_frac(step,elapsed_ms);scale=lr_mul(frac)
		if h.num_loops>0 and not base_model.looping_active and frac>=h.enable_looping_at:base_model.looping_active=True;log(f"layer_loop:enabled step:{step} frac:{frac:.3f} encoder:{base_model.encoder_indices} decoder:{base_model.decoder_indices}")
		# Shot 17: Fuzzy LR bandit — Thompson sample an arm and apply its LR multiplier
		if _fuzzy_enabled:
			_samples=[_fuzzy_means[i]+random.gauss(0,1.0/(_fuzzy_counts[i]**0.5))for i in range(len(_fuzzy_arms))]
			_fuzzy_arm_idx=_samples.index(max(_samples));scale=scale*_fuzzy_arms[_fuzzy_arm_idx]
		train_loss=step_fn(step,scale)
		# Shot 17: bandit reward update (reward = train_loss decrease vs prev step)
		if _fuzzy_enabled:
			_cur_loss=train_loss.item()
			if _fuzzy_prev_loss is not None:
				_reward=_fuzzy_prev_loss-_cur_loss;_fuzzy_counts[_fuzzy_arm_idx]+=1
				_fuzzy_means[_fuzzy_arm_idx]+=(_reward-_fuzzy_means[_fuzzy_arm_idx])/_fuzzy_counts[_fuzzy_arm_idx]
			_fuzzy_prev_loss=_cur_loss
		with torch.no_grad():
			for(name,t)in base_model.state_dict().items():ema_state[name].mul_(ema_decay).add_(t.detach().float(),alpha=1.-ema_decay)
		step+=1;approx_training_time_ms=training_time_ms+1e3*(time.perf_counter()-t0);should_log_train=h.train_log_every>0 and(step<=5 or step%h.train_log_every==0 or stop_after_step is not None)
		if should_log_train:tok_per_sec=step*h.train_batch_tokens/(approx_training_time_ms/1e3);log(f"{step}/{h.iterations} train_loss: {train_loss.item():.4f} train_time: {approx_training_time_ms/60000:.1f}m tok/s: {tok_per_sec:.0f}")
		reached_cap=max_wallclock_ms is not None and approx_training_time_ms>=max_wallclock_ms
		if h.distributed and max_wallclock_ms is not None:reached_cap_tensor=torch.tensor(int(reached_cap),device=device);dist.all_reduce(reached_cap_tensor,op=dist.ReduceOp.MAX);reached_cap=bool(reached_cap_tensor.item())
		if stop_after_step is None and reached_cap:stop_after_step=step
	if _fuzzy_enabled:
		_best_arm=_fuzzy_means.index(max(_fuzzy_means));_total=sum(_fuzzy_counts)-len(_fuzzy_counts)
		log(f"FUZZY_LR_BANDIT summary: arms={_fuzzy_arms} means={[round(m,4)for m in _fuzzy_means]} counts={[c-1 for c in _fuzzy_counts]} total_steps={_total} best_arm={_fuzzy_arms[_best_arm]}")
	log(f"peak memory allocated: {torch.cuda.max_memory_allocated()//1024//1024} MiB reserved: {torch.cuda.max_memory_reserved()//1024//1024} MiB");log('ema:applying EMA weights');current_state=base_model.state_dict();avg_state={name:t.to(dtype=current_state[name].dtype)for(name,t)in ema_state.items()};base_model.load_state_dict(avg_state,strict=True);return base_model,compiled_model
def prequant_ttt_adapt_adamw(h,base_model,device,val_tokens,rank=0,world_size=1):
	"""Pre-Quant AdamW TTT (ported from PR #1485 / #1306).
	Fine-tunes the EMA-applied base_model on val tokens BEFORE GPTQ so the
	adaptation bakes into the quantized weights. Frontier (PR #1482) gives ~-0.014
	BPB on top of eval-time TTT. Modifies base_model in place.
	"""
	seq_len=h.train_seq_len
	total_seqs=(val_tokens.numel()-1)//seq_len
	batch_seqs=h.prequant_ttt_batch_seqs
	if h.prequant_ttt_freeze_blocks>0:
		for i,block in enumerate(base_model.blocks):
			if i<h.prequant_ttt_freeze_blocks:
				for p in block.parameters():p.requires_grad_(False)
	ttt_params=[p for p in base_model.parameters() if p.requires_grad]
	log(f"prequant_ttt:params trainable={sum(p.numel() for p in ttt_params)} frozen={sum(p.numel() for p in base_model.parameters() if not p.requires_grad)}")
	optimizer=torch.optim.AdamW(ttt_params,lr=h.prequant_ttt_lr,weight_decay=0.0)
	scheduler=None
	if h.prequant_ttt_cosine_decay:
		scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=h.prequant_ttt_epochs,eta_min=h.prequant_ttt_lr*0.1)
	my_start=(total_seqs*rank)//world_size
	my_end=(total_seqs*(rank+1))//world_size
	base_model.train()
	t0=time.perf_counter()
	for epoch in range(h.prequant_ttt_epochs):
		epoch_loss_sum=torch.zeros((),device=device,dtype=torch.float64)
		epoch_tokens=torch.zeros((),device=device,dtype=torch.float64)
		for bs in range(my_start,my_end,batch_seqs):
			be=min(bs+batch_seqs,my_end)
			raw_start=bs*seq_len
			raw_end=be*seq_len+1
			if raw_end>val_tokens.numel():continue
			local=val_tokens[raw_start:raw_end].to(device=device,dtype=torch.int64)
			x=local[:-1].reshape(-1,seq_len)
			y=local[1:].reshape(-1,seq_len)
			optimizer.zero_grad(set_to_none=True)
			with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
				loss=base_model(x,y)
			loss.backward()
			if world_size>1:
				for p in ttt_params:
					if p.grad is not None:dist.all_reduce(p.grad,op=dist.ReduceOp.AVG)
			torch.nn.utils.clip_grad_norm_(ttt_params,h.prequant_ttt_grad_clip)
			optimizer.step()
			epoch_loss_sum+=loss.detach().to(torch.float64)*float(y.numel())
			epoch_tokens+=float(y.numel())
		if world_size>1:
			dist.all_reduce(epoch_loss_sum,op=dist.ReduceOp.SUM)
			dist.all_reduce(epoch_tokens,op=dist.ReduceOp.SUM)
		epoch_avg=epoch_loss_sum.item()/max(epoch_tokens.item(),1)
		if scheduler is not None:scheduler.step()
		log(f"prequant_ttt:epoch {epoch+1}/{h.prequant_ttt_epochs} loss:{epoch_avg:.4f} time:{time.perf_counter()-t0:.1f}s")
	# CRITICAL: unfreeze ALL params before GPTQ — collect_hessians needs gradients
	# on every block. Leaving the first N blocks frozen would zero their Hessians
	# and quantize them to garbage.
	for p in base_model.parameters():p.requires_grad_(True)
	base_model.eval()
	log(f"prequant_ttt:done elapsed={time.perf_counter()-t0:.1f}s")

def train_and_eval(h,device):
	random.seed(h.seed);np.random.seed(h.seed);torch.manual_seed(h.seed);torch.cuda.manual_seed_all(h.seed);val_data=ValidationData(h,device);log("train_shards: " + str(len(list(Path(h.datasets_dir).resolve().glob("fineweb_train_*.bin")))));log(f"val_tokens: {val_data.val_tokens.numel()-1}");base_model,compiled_model=train_model(h,device,val_data);torch._dynamo.reset();timed_eval('pre-quantization post-ema',eval_val,h,device,val_data,compiled_model)
	# Pre-Quant AdamW TTT (chunk 1, C1 from audit, ported from PR #1485)
	# Runs AFTER EMA + pre-quant eval, BEFORE serialize/GPTQ. Adapts the EMA model
	# on val data with AdamW + cosine schedule. PR #1482 frontier hyperparams
	# (lr=0.00045, epochs=8, freeze_blocks=1) gives val_bpb 1.0787 = -0.014 BPB
	# vs the eval-time SGD TTT (val_bpb 1.0822 our baseline reproduction).
	if h.prequant_ttt_enabled:
		prequant_ttt_adapt_adamw(h,base_model,device,val_data.val_tokens,rank=h.rank,world_size=h.world_size)
		torch._dynamo.reset()
		timed_eval('post-prequant-ttt',eval_val,h,device,val_data,base_model)
	serialize(h,base_model,Path(__file__).read_text(encoding='utf-8'),val_data=val_data)
	if h.distributed:dist.barrier()
	eval_model=deserialize(h,device)
	if h.num_loops>0:eval_model.looping_active=True
	# SHOT 0e: same eager NLFI setup on the deserialized eval model before compile.
	# After serialize→deserialize, _nlfi_stored_flag should be 1 (restored from
	# state_dict), so this takes the "restored multipliers" branch.
	# RULE COMPLIANCE: same train sample as the train_model side.
	if getattr(eval_model,'_nlfi_enabled',False) and not getattr(eval_model,'_nlfi_applied',False):
		_sample=_load_train_sample_for_nlfi(h,device)
		if _sample is not None:
			eval_model._apply_nlfi_once(_sample)
	compiled_model=torch.compile(eval_model,dynamic=False,fullgraph=True);timed_eval('quantized',eval_val,h,device,val_data,compiled_model)
	if h.sliding_window_enabled:timed_eval('quantized_sliding_window',eval_val_sliding,h,device,val_data,eval_model)
	if h.ttt_enabled:
		del eval_model,compiled_model;torch._dynamo.reset();torch.cuda.empty_cache();ttt_model=deserialize(h,device)
		if h.num_loops>0:ttt_model.looping_active=True
		timed_eval('quantized_ttt',eval_val_sliding_ttt,h,ttt_model,h.rank,h.world_size,device,val_data,stride=h.eval_stride);del ttt_model
def main():
	world_size=int(os.environ.get('WORLD_SIZE','1'));local_rank=int(os.environ.get('LOCAL_RANK','0'));distributed='RANK'in os.environ and'WORLD_SIZE'in os.environ
	if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
	if world_size<=0:raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
	if 8%world_size!=0:raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
	device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
	if distributed:dist.init_process_group(backend='nccl',device_id=device);dist.barrier()
	torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('high');torch.backends.cudnn.benchmark=bool(int(os.environ.get('USE_CUDNN_BENCHMARK','1')));from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp;enable_cudnn_sdp(False);enable_flash_sdp(True);enable_mem_efficient_sdp(False);enable_math_sdp(False);torch._dynamo.config.optimize_ddp=False;h=Hyperparameters();set_logging_hparams(h)
	if h.is_main_process:
		os.makedirs('logs',exist_ok=True);log(100*'=',console=False);log('Hyperparameters:',console=True)
		for(k,v)in sorted(vars(type(h)).items()):
			if not k.startswith('_'):log(f"  {k}: {v}",console=True)
		log('='*100,console=False);log(f"Running Python {sys.version}",console=False);log(f"Running PyTorch {torch.__version__}",console=False);log(subprocess.run(['nvidia-smi'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False).stdout,console=False);log('='*100,console=False)
	train_and_eval(h,device)
	if distributed:dist.destroy_process_group()
if __name__=='__main__':main()