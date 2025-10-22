// Simple client-side router and UI logic
(function() {
	const app = document.getElementById('app');

	const state = {
		route: 'home',
		ticketSubmitted: false,
		drawingDataUrl: null,
		pastGenerated: [],
		seedFromUpload: null,
		rngRun: null, // { runDir, keyPath, numbers }
		pastRun: null, // { runDir, keyPath, numbers }
		randomnessKeyPath: null
	};

	function setActiveNav(route) {
		for (const a of document.querySelectorAll('.nav-link')) {
			a.classList.toggle('active', a.dataset.route === route);
		}
	}

	function navigate(route) {
		state.route = route;
		setActiveNav(route);
		render();
	}

	function render() {
		if (state.route === 'home') return renderHome();
		if (state.route === 'upload') return renderUpload();
		if (state.route === 'past') return renderPast();
		if (state.route === 'randomness') return renderRandomness();
		if (state.route === 'rng') return renderRNG();
	}

	function renderHome() {
		app.innerHTML = `
			<section class="screen">
				<div class="hero">
					<div class="tile ${state.ticketSubmitted ? 'disabled' : ''}">
						<div>
							<div class="title">Загрузить билет</div>
							<div class="subtitle">Добавьте номер и рисунок для участия</div>
						</div>
						<a class="cta" ${state.ticketSubmitted ? 'aria-disabled="true" tabindex="-1"' : ''} data-goto="upload" href="#">${state.ticketSubmitted ? 'Отправлено' : 'Открыть'}</a>
					</div>
					<div class="tile">
						<div>
							<div class="title">Прошлые тиражи</div>
							<div class="subtitle">История и проверка случайности</div>
						</div>
						<a class="cta" data-goto="past" href="#">Открыть</a>
					</div>
					<div class="tile">
						<div>
							<div class="title">Генератор случайных чисел</div>
							<div class="subtitle">Создание комбинации из рисунка</div>
						</div>
						<a class="cta" data-goto="rng" href="#">Открыть</a>
					</div>
				</div>
				<div class="footer-note">Результаты в 13:00</div>
			</section>
		`;

		app.querySelectorAll('[data-goto]').forEach(el => {
			el.addEventListener('click', (e) => {
				e.preventDefault();
				navigate(el.getAttribute('data-goto'));
			});
		});
	}

	function renderUpload() {
		app.innerHTML = `
			<section class="screen">
				<div class="section">
					<h2>Загрузка билета</h2>
					<div class="row two">
						<div>
							<label>Номер билета</label>
							<input id="ticketInput" type="text" placeholder="Например: 1234567890" />
							<div style="height:12px"></div>
							<label>Палитра</label>
							<div class="palette" id="palette"></div>
							<div style="height:12px"></div>
							<label>Толщина кисти</label>
							<div class="thickness-controls">
								<input type="range" id="uploadBrushThickness" class="thickness-slider" min="2" max="30" value="10" />
								<span class="thickness-value" id="uploadBrushValue">10px</span>
							</div>
							<div style="height:8px"></div>
							<button class="btn" id="eraserBtn">Ластик</button>
						</div>
						<div>
							<label>Рисунок</label>
							<div class="canvas-wrap">
								<canvas id="drawCanvas" width="800" height="340"></canvas>
							</div>
						</div>
					</div>
					<div style="height:12px"></div>
					<div style="display:flex; gap:8px; justify-content:flex-end;">
						<button class="btn" data-goto="home">Отмена</button>
						<button class="btn primary" id="submitDrawing">Отправить рисунок</button>
					</div>
				</div>
			</section>
		`;

		app.querySelector('[data-goto="home"]').addEventListener('click', (e) => { e.preventDefault(); navigate('home'); });

		// Palette
		const palette = app.querySelector('#palette');
		const colors = ['#ffffff','#ffd200','#ff6a00','#ff3060','#00e0ff','#00ff99','#7b61ff','#1f2937','#94a3b8'];
		let current = colors[0];
		colors.forEach((c, i) => {
			const s = document.createElement('button');
			s.type = 'button';
			s.className = 'swatch' + (i === 0 ? ' active' : '');
			s.style.background = c;
			s.addEventListener('click', () => {
				current = c;
				palette.querySelectorAll('.swatch').forEach(x => x.classList.remove('active'));
				s.classList.add('active');
				isErasing = false;
				// Deactivate eraser button
				app.querySelector('#eraserBtn').classList.remove('active');
			});
			palette.appendChild(s);
		});

		let isDrawing = false;
		let isErasing = false;
		const canvas = app.querySelector('#drawCanvas');
		const ctx = canvas.getContext('2d');
		ctx.fillStyle = '#0b0b0b';
		ctx.fillRect(0,0,canvas.width,canvas.height);
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
		ctx.lineWidth = 10;

		// Thickness control for upload screen
		const uploadBrushThickness = app.querySelector('#uploadBrushThickness');
		const uploadBrushValue = app.querySelector('#uploadBrushValue');
		uploadBrushThickness.addEventListener('input', (e) => {
			const value = e.target.value;
			uploadBrushValue.textContent = value + 'px';
			ctx.lineWidth = parseInt(value);
		});

		function pos(e){
			const r = canvas.getBoundingClientRect();
			if (e.touches && e.touches[0]) {
				return {x:(e.touches[0].clientX - r.left) * (canvas.width/r.width), y:(e.touches[0].clientY - r.top) * (canvas.height/r.height)};
			}
			return {x:(e.clientX - r.left) * (canvas.width/r.width), y:(e.clientY - r.top) * (canvas.height/r.height)};
		}

		function drawLine(from, to) {
			if (isErasing) {
				ctx.globalCompositeOperation = 'destination-out';
				ctx.strokeStyle = 'rgba(0,0,0,1)';
			} else {
				ctx.globalCompositeOperation = 'source-over';
				ctx.strokeStyle = current;
			}
			ctx.beginPath();
			ctx.moveTo(from.x, from.y);
			ctx.lineTo(to.x, to.y);
			ctx.stroke();
		}

		let last = null;
		function start(e){ isDrawing = true; last = pos(e); }
		function move(e){ if(!isDrawing) return; const p = pos(e); drawLine(last, p); last = p; }
		function end(){ isDrawing = false; last = null; }

		canvas.addEventListener('mousedown', start);
		canvas.addEventListener('mousemove', move);
		window.addEventListener('mouseup', end);
		canvas.addEventListener('touchstart', (e)=>{ e.preventDefault(); start(e); }, {passive:false});
		canvas.addEventListener('touchmove', (e)=>{ e.preventDefault(); move(e); }, {passive:false});
		canvas.addEventListener('touchend', (e)=>{ e.preventDefault(); end(); }, {passive:false});

		app.querySelector('#eraserBtn').addEventListener('click', ()=>{ 
			isErasing = !isErasing; 
			const eraserBtn = app.querySelector('#eraserBtn');
			if (isErasing) {
				eraserBtn.classList.add('active');
				palette.querySelectorAll('.swatch').forEach(x => x.classList.remove('active'));
			} else {
				eraserBtn.classList.remove('active');
				// Reactivate the first color
				palette.querySelector('.swatch').classList.add('active');
			}
		});

		app.querySelector('#submitDrawing').addEventListener('click', () => {
			const ticket = /** @type {HTMLInputElement} */(app.querySelector('#ticketInput')).value.trim();
			if (!ticket) { alert('Введите номер билета'); return; }
			state.drawingDataUrl = canvas.toDataURL('image/png');
			state.ticketSubmitted = true;
			navigate('home');
		});
	}

	function renderPast() {
		app.innerHTML = `
			<section class="screen">
				<div class="section">
					<h2>Прошлые тиражи</h2>
					<div class="row two">
						<div>
							<label>Дата</label>
							<input id="datePicker" type="date" />
							<div style="height:8px"></div>
							<button class="btn" id="loadAggregate">Загрузить общую картину</button>
							<div style="height:12px"></div>
							<label>Комбинация из 60 чисел</label>
							<div id="sixty" class="numbers"></div>
						</div>
						<div>
							<label>Загрузка собственного фото</label>
							<input id="photoInput" type="file" accept="image/*" />
							<div style="height:8px"></div>
							<button class="btn primary" id="genFromPhoto">Генерация комбинации</button>
							<div style="height:12px"></div>
							<div class="viz-box" id="viz">Окно визуализации процесса генерации…</div>
							<div style="height:8px"></div>
							<label>Сгенерированная комбинация</label>
							<div id="generated" class="numbers"></div>
							<div style="height:12px"></div>
							<button class="btn" id="checkRandom">Проверка на случайность</button>
						</div>
					</div>
				</div>
			</section>
		`;

		// Populate 60 numbers (dummy) deterministically from date
		const dateEl = app.querySelector('#datePicker');
		const out60 = app.querySelector('#sixty');
		const viz = app.querySelector('#viz');
		const genOut = app.querySelector('#generated');

		function seedFromDate(str){
			let h = 2166136261 >>> 0;
			for (let i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
			return h >>> 0;
		}
		function mulberry32(a){ return function(){ a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t ^= t + Math.imul(t ^ (t >>> 7), 61 | t); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
		function generate60(seed){ const rnd = mulberry32(seed); const arr=[]; for(let i=0;i<60;i++){ arr.push(1+Math.floor(rnd()*99)); } return arr; }

		function renderBadges(arr, target){ target.innerHTML=''; arr.forEach(n => { const b=document.createElement('span'); b.className='badge accent'; b.textContent=String(n); target.appendChild(b); }); }

		dateEl.addEventListener('change', () => {
			const s = seedFromDate(dateEl.value || new Date().toISOString().slice(0,10));
			renderBadges(generate60(s), out60);
		});
		dateEl.valueAsDate = new Date();
		dateEl.dispatchEvent(new Event('change'));

		app.querySelector('#loadAggregate').addEventListener('click', ()=>{
			viz.textContent = 'Загружается агрегированная картина пользователей… (демо)';
			setTimeout(()=>{ viz.textContent = 'Показана синтетическая тепловая карта вводов (демо).'; }, 800);
		});

		app.querySelector('#genFromPhoto').addEventListener('click', async ()=>{
			const file = /** @type {HTMLInputElement} */(app.querySelector('#photoInput')).files?.[0];
			if (!file) { alert('Загрузите фото'); return; }
			viz.textContent = 'Извлечение энтропии из изображения…';
			const buf = await file.arrayBuffer();
			const seed = hashBytes(new Uint8Array(buf));
			const numbers = generate60(seed);
			renderBadges(numbers, genOut);
			state.pastGenerated = numbers;
			viz.textContent = 'Энтропия извлечена, комбинация сгенерирована.';
		});

		app.querySelector('#checkRandom').addEventListener('click', ()=>{
			navigate('randomness');
		});
	}

	function renderRandomness(){
		app.innerHTML = `
			<section class="screen">
				<div class="section">
					<h2>Проверка на случайность (базовые тесты)</h2>
					<div class="row">
						<div class="viz-box" id="rndReport">Выполняется анализ…</div>
						<div style="display:flex; gap:8px; justify-content:flex-end;">
							<button class="btn" data-goto="past">Назад</button>
						</div>
					</div>
				</div>
			</section>
		`;
		app.querySelector('[data-goto="past"]').addEventListener('click', (e)=>{ e.preventDefault(); navigate('past'); });
		const reportEl = app.querySelector('#rndReport');
		const arr = state.pastGenerated && state.pastGenerated.length ? state.pastGenerated : demoSequence();
		const res = runBasicTests(arr);
		reportEl.textContent = formatReport(res);
	}

	// Utilities
	function hashBytes(bytes){
		let h = 2166136261 >>> 0;
		for (let i=0;i<bytes.length;i++){ h ^= bytes[i]; h = Math.imul(h, 16777619); }
		return h >>> 0;
	}
	function demoSequence(){ const rnd = Math.random; const a=[]; for(let i=0;i<60;i++){ a.push(1+Math.floor(rnd()*99)); } return a; }

	function runBasicTests(nums){
		// Frequency test (chi-square against uniform 1..99 buckets folded to 10 bins)
		const bins = new Array(10).fill(0);
		nums.forEach(n => { bins[Math.floor((n-1)/10)]++; });
		const expected = nums.length / 10;
		let chi2 = 0;
		for (let i=0;i<10;i++){ const o=bins[i]; chi2 += (o-expected)*(o-expected)/expected; }
		// Runs test on up/down relative to median
		const median = 50; // approximate
		let runs = 1;
		for (let i=1;i<nums.length;i++){
			if ((nums[i] > median) !== (nums[i-1] > median)) runs++;
		}
		// Serial correlation
		let sum=0, sumSq=0, sumLag=0;
		for (let i=0;i<nums.length;i++){ sum += nums[i]; sumSq += nums[i]*nums[i]; if(i) sumLag += nums[i]*nums[i-1]; }
		const n = nums.length;
		const corr = (n*sumLag - sum* (sum - nums[0])) / Math.sqrt((n*sumSq - sum*sum) * (n*sumSq - sum*sum));
		return { bins, chi2, runs, corr };
	}
	function formatReport(r){
		return [
			`Частоты по 10 корзинам: ${r.bins.join(', ')}`,
			`Хи-квадрат: ${r.chi2.toFixed(2)} (чем ближе к 9±, тем лучше для 10 бинов)`,
			`Число серий (runs) относительно медианы: ${r.runs} (ожидаемо ~ ${Math.round(1 + (2*(60-1))/2)})`,
			`Сериальная корреляция: ${Number.isFinite(r.corr)? r.corr.toFixed(3): 'NaN'}`
		].join('\n');
	}

	function renderRNG() {
		app.innerHTML = `
			<section class="screen">
				<div class="section">
					<h2>Генератор случайных чисел из рисунка</h2>
					<div class="row two">
						<div>
							<div class="control-group">
								<label>Палитра</label>
								<div class="palette" id="rngPalette"></div>
							</div>
							
							<div class="control-group">
								<label>Толщина кисти</label>
								<div class="thickness-controls">
									<input type="range" id="brushThickness" class="thickness-slider" min="2" max="30" value="10" />
									<span class="thickness-value" id="brushValue">10px</span>
								</div>
								<button class="btn" id="rngEraserBtn">Ластик</button>
							</div>
							
							<div class="control-group">
								<label>Диапазон чисел</label>
								<div class="number-input-group">
									<input type="number" id="minTextInput" class="number-input" value="0" min="0" max="9999999999" placeholder="Мин" />
									<span style="color: var(--muted); font-weight: 600;">до</span>
									<input type="number" id="maxTextInput" class="number-input" value="100" min="0" max="9999999999" placeholder="Макс" />
								</div>
							</div>
							
							<div class="control-group">
								<label>Количество чисел в комбинации</label>
								<input type="number" id="countTextInput" class="number-input" value="6" min="1" max="50" placeholder="Количество" style="width: 120px;" />
							</div>
							
							<div class="control-group">
								<button class="btn primary" id="generateFromDrawing" style="width: 100%; padding: 14px; font-size: 16px;">Сгенерировать комбинацию</button>
							</div>
						</div>
						
						<div>
							<div class="control-group">
								<label>Рисунок</label>
								<div class="canvas-wrap">
									<canvas id="rngCanvas" width="800" height="340"></canvas>
								</div>
							</div>
							
							<div class="control-group">
								<label>Сгенерированная комбинация</label>
								<div id="rngResult" class="numbers"></div>
								<div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
									<button class="btn" id="rngZipBtn" disabled>Скачать архив</button>
									<button class="btn" id="rngRandomnessBtn" disabled>Проверка на случайность</button>
								</div>
							</div>
						</div>
					</div>
					
					<div style="margin-top: 32px; display:flex; gap:12px; justify-content:flex-end;">
						<button class="btn" data-goto="home">Назад</button>
					</div>
				</div>
			</section>
		`;

		app.querySelector('[data-goto="home"]').addEventListener('click', (e) => { e.preventDefault(); navigate('home'); });

		// Palette for RNG screen
		const palette = app.querySelector('#rngPalette');
		const colors = ['#ffffff','#ffd200','#ff6a00','#ff3060','#00e0ff','#00ff99','#7b61ff','#1f2937','#94a3b8'];
		let currentColor = colors[0];
		colors.forEach((c, i) => {
			const s = document.createElement('button');
			s.type = 'button';
			s.className = 'swatch' + (i === 0 ? ' active' : '');
			s.style.background = c;
			s.addEventListener('click', () => {
				currentColor = c;
				palette.querySelectorAll('.swatch').forEach(x => x.classList.remove('active'));
				s.classList.add('active');
				isErasing = false;
				// Deactivate eraser button
				app.querySelector('#rngEraserBtn').classList.remove('active');
			});
			palette.appendChild(s);
		});

		let isDrawing = false;
		let isErasing = false;
		const canvas = app.querySelector('#rngCanvas');
		const ctx = canvas.getContext('2d');
		ctx.fillStyle = '#0b0b0b';
		ctx.fillRect(0,0,canvas.width,canvas.height);
		ctx.lineCap = 'round';
		ctx.lineJoin = 'round';
		ctx.lineWidth = 10;

		// Thickness control
		const brushThickness = app.querySelector('#brushThickness');
		const brushValue = app.querySelector('#brushValue');
		brushThickness.addEventListener('input', (e) => {
			const value = e.target.value;
			brushValue.textContent = value + 'px';
			ctx.lineWidth = parseInt(value);
		});

		// Text inputs only
		const minTextInput = app.querySelector('#minTextInput');
		const maxTextInput = app.querySelector('#maxTextInput');
		const countTextInput = app.querySelector('#countTextInput');

		// Text input validation and sync
		minTextInput.addEventListener('input', (e) => {
			const value = Math.max(0, Math.min(9999999999, parseInt(e.target.value) || 0));
			minTextInput.value = value;
			// Ensure min doesn't exceed max
			if (value >= parseInt(maxTextInput.value)) {
				const newMax = Math.min(9999999999, value + 1);
				maxTextInput.value = newMax;
			}
		});

		maxTextInput.addEventListener('input', (e) => {
			const value = Math.max(0, Math.min(9999999999, parseInt(e.target.value) || 100));
			maxTextInput.value = value;
			// Ensure max doesn't go below min
			if (value <= parseInt(minTextInput.value)) {
				const newMin = Math.max(0, value - 1);
				minTextInput.value = newMin;
			}
		});

		countTextInput.addEventListener('input', (e) => {
			const value = Math.max(1, Math.min(50, parseInt(e.target.value) || 6));
			countTextInput.value = value;
		});

		function getPos(e){
			const r = canvas.getBoundingClientRect();
			if (e.touches && e.touches[0]) {
				return {x:(e.touches[0].clientX - r.left) * (canvas.width/r.width), y:(e.touches[0].clientY - r.top) * (canvas.height/r.height)};
			}
			return {x:(e.clientX - r.left) * (canvas.width/r.width), y:(e.clientY - r.top) * (canvas.height/r.height)};
		}

		function drawLine(from, to) {
			if (isErasing) {
				ctx.globalCompositeOperation = 'destination-out';
				ctx.strokeStyle = 'rgba(0,0,0,1)';
			} else {
				ctx.globalCompositeOperation = 'source-over';
				ctx.strokeStyle = currentColor;
			}
			ctx.beginPath();
			ctx.moveTo(from.x, from.y);
			ctx.lineTo(to.x, to.y);
			ctx.stroke();
		}

		let lastPos = null;
		function startDraw(e){ isDrawing = true; lastPos = getPos(e); }
		function moveDraw(e){ if(!isDrawing) return; const p = getPos(e); drawLine(lastPos, p); lastPos = p; }
		function endDraw(){ isDrawing = false; lastPos = null; }

		canvas.addEventListener('mousedown', startDraw);
		canvas.addEventListener('mousemove', moveDraw);
		window.addEventListener('mouseup', endDraw);
		canvas.addEventListener('touchstart', (e)=>{ e.preventDefault(); startDraw(e); }, {passive:false});
		canvas.addEventListener('touchmove', (e)=>{ e.preventDefault(); moveDraw(e); }, {passive:false});
		canvas.addEventListener('touchend', (e)=>{ e.preventDefault(); endDraw(); }, {passive:false});

		app.querySelector('#rngEraserBtn').addEventListener('click', ()=>{ 
			isErasing = !isErasing; 
			const eraserBtn = app.querySelector('#rngEraserBtn');
			if (isErasing) {
				eraserBtn.classList.add('active');
				palette.querySelectorAll('.swatch').forEach(x => x.classList.remove('active'));
			} else {
				eraserBtn.classList.remove('active');
				// Reactivate the first color
				palette.querySelector('.swatch').classList.add('active');
			}
		});

		app.querySelector('#generateFromDrawing').addEventListener('click', async () => {
			const min = parseInt(minTextInput.value) || 0;
			const max = parseInt(maxTextInput.value) || 100;
			const count = parseInt(countTextInput.value) || 6;
			
			if (min >= max) { alert('Минимальное значение должно быть меньше максимального'); return; }
			if (count > (max - min + 1)) { alert('Количество чисел не может быть больше диапазона'); return; }

			// Show loading state
			const generateBtn = app.querySelector('#generateFromDrawing');
			const originalText = generateBtn.textContent;
			generateBtn.textContent = 'Генерация...';
			generateBtn.disabled = true;

			try {
				// 1) Save canvas to server to get photos folder
				const canvasDataUrl = canvas.toDataURL('image/jpeg', 0.92);
				const storeResp = await fetch('http://localhost:8000/store-canvas', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ canvas_data: canvasDataUrl })
				});
				if (!storeResp.ok) throw new Error('store-canvas failed');
				const storeData = await storeResp.json();

				// 2) Generate numbers using photos_folder
				const genResp = await fetch('http://localhost:8000/generate', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({
						photos_folder: storeData.photos_folder,
						low: min,
						high: max,
						count: count
					})
				});
				if (!genResp.ok) throw new Error('generate failed');
				const genData = await genResp.json();

				// Save run info for randomness check and zip
				state.rngRun = { runDir: genData.run_dir, keyPath: genData.key, numbers: genData.numbers || [] };
				state.randomnessKeyPath = genData.key;

				const resultEl = app.querySelector('#rngResult');
				resultEl.innerHTML = '';
				if (count > 50) {
					// Too many — show only archive button
					resultEl.innerHTML = '<span class="note">Сгенерировано ' + count + ' чисел. Скачайте архив для просмотра.</span>';
				} else {
					(genData.numbers || []).forEach(n => {
						const b = document.createElement('span');
						b.className = 'badge accent';
						b.textContent = String(n);
						resultEl.appendChild(b);
					});
				}

				// Enable buttons
				const zipBtn = app.querySelector('#rngZipBtn');
				zipBtn.disabled = false;
				zipBtn.onclick = () => downloadArchive(genData.run_dir);
				const rndBtn = app.querySelector('#rngRandomnessBtn');
				rndBtn.disabled = false;
				rndBtn.onclick = () => runNistAndShow(genData.key, 'rng');

			} catch (error) {
				console.error('Error generating numbers:', error);
				// Fallback to client-side generation
				generateClientSide(min, max, count);
			} finally {
				// Restore button state
				generateBtn.textContent = originalText;
				generateBtn.disabled = false;
			}
		});

		// Fallback client-side generation
		function generateClientSide(min, max, count) {
			// Extract entropy from canvas
			const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
			const pixels = imageData.data;
			
			// Simple hash from pixel data
			let seed = 2166136261 >>> 0;
			for (let i = 0; i < pixels.length; i += 4) {
				seed ^= pixels[i] ^ pixels[i+1] ^ pixels[i+2];
				seed = Math.imul(seed, 16777619);
			}
			seed = seed >>> 0;

			// Generate numbers using seeded PRNG
			const rng = mulberry32(seed);
			const result = [];
			const used = new Set();
			
			while (result.length < count) {
				const num = min + Math.floor(rng() * (max - min + 1));
				if (!used.has(num)) {
					used.add(num);
					result.push(num);
				}
			}
			
			const resultEl = app.querySelector('#rngResult');
			resultEl.innerHTML = '';
			if (count > 50) {
				resultEl.innerHTML = '<span class="note">Сгенерировано ' + count + ' чисел. Скачайте архив для просмотра.</span>';
			} else {
				result.forEach(n => {
					const b = document.createElement('span');
					b.className = 'badge accent';
					b.textContent = String(n);
					resultEl.appendChild(b);
				});
			}
		}

		async function downloadArchive(runDir) {
			try {
				const resp = await fetch('http://localhost:8000/archive-run', {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ run_dir: runDir })
				});
				if (!resp.ok) throw new Error('archive-run failed');
				const blob = await resp.blob();
				const url = URL.createObjectURL(blob);
				const a = document.createElement('a');
				a.href = url; a.download = 'run.zip';
				document.body.appendChild(a); a.click(); a.remove();
				URL.revokeObjectURL(url);
			} catch (e) { console.error(e); }
		}

		async function runNistAndShow(keyPath, source) {
			try {
				const resp = await fetch('http://localhost:8000/nist/generate-from-key', {
					method: 'POST', headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify({ key_path: keyPath, count: 1000000, low: 0, high: 1 })
				});
				if (!resp.ok) throw new Error('nist generate-from-key failed');
				const data = await resp.json();
				alert('NIST: ' + (data.passed_of_5 || 0) + ' / 5 тестов пройдено');
			} catch (e) { console.error(e); }
		}

		// Helper function for seeded random
		function mulberry32(a) {
			return function() {
				a |= 0; a = (a + 0x6D2B79F5) | 0;
				let t = Math.imul(a ^ (a >>> 15), 1 | a);
				t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
				return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
			};
		}
	}

	// Nav bindings
	document.querySelectorAll('.nav-link').forEach(a => {
		a.addEventListener('click', (e)=>{ e.preventDefault(); navigate(a.dataset.route); });
	});

	render();
})();



