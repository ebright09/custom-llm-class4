'use strict';
const $ = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = n => n === null ? 'Not scorable' : `${(n * 100).toFixed(1)}%`;
const num = n => Number(n).toLocaleString('en-US');
const names = ['What we fed the raccoon.', 'Words get employee numbers.', 'Mandatory professional development.', 'The performance review.', 'Promoted anyway.'];
const examples = {
  goose: {quip:'The cup has been reclassified. Please update your records.', why:'This sentence invites a description of the cup. The model instead generated a category phrase: “a goose.” It assembled a familiar-looking pattern without keeping the meaning consistent. That observable failure does not tell us exactly which internal weights caused it.'},
  spatial: {quip:'Located the furniture. Has requested a facilities budget.', why:'The model completed the inverse relationship: if the clock is above the sofa, the sofa is below the clock. This exact pair occurs in its teaching data, so this example can show recall. The fixed lamp/desk test checks different objects.'},
  opposite: {quip:'A bold commitment to being wrong in both directions.', why:'The model repeated “big” instead of giving an opposite. The teaching corpus does include big/small, so even a taught relationship is unreliable in this sampled continuation. This is not the same measurement as the four-choice test score.'},
  customer: {quip:'An entire customer journey. Still no refund.', why:'The model learned repeated customer-and-product sentence patterns. A plausible continuation shows that narrow pattern learning; it does not establish an understanding of customers.'},
  homework: {quip:'Your request has been escalated to a different dumpster.', why:'Several prompt words are absent from this model’s vocabulary and become unknown tokens. It also was not trained to follow requests. The result is a continuation of text, not an answer from an assistant.'}
};
let data, step = 0, lessonModel = 'starter', sampleStep = '0', vectorStage = 'after', evalStage = 'final', selectedCase = 0;
let renderedStep = -1;
const turns = [];
const exp = () => data.experiments[lessonModel];
const evidence = (experiment, file, label) => `<a class="source-link" href="/evidence/llm_runs/${data.experiments[experiment].run}/${file}">${esc(label)} ↗</a>`;
const modelPicker = () => `<label>Evidence from<select id="lesson-model"><option value="starter" ${lessonModel==='starter'?'selected':''}>Starter experiment</option><option value="expanded" ${lessonModel==='expanded'?'selected':''}>Expanded experiment</option></select></label>`;
const tryButton = (key, label='Try this in the playground →') => `<button class="lesson-action" data-try="${key}">${label}</button>`;

function lesson() {
  const sameStep = renderedStep === step;
  const openDetails = sameStep ? [...$('lesson-body').querySelectorAll('details')].map(d=>d.open) : [];
  const focused = document.activeElement;
  const focusSelector = sameStep && $('lesson-body').contains(focused)
    ? (focused.id ? `#${focused.id}` : focused.dataset.vector ? `[data-vector="${focused.dataset.vector}"]`
      : focused.dataset.milestone ? `[data-milestone="${focused.dataset.milestone}"]` : null) : null;
  const e = exp();
  const i = e.inspection;
  let content = '';
  if (step === 0) {
    const a = data.experiments.starter, b = data.experiments.expanded;
    content = `<p class="lesson-intro">A <b>corpus</b> is a collection of teaching examples. The model practices guessing the next word. What you feed it shapes what it can learn.</p>
      <div class="two-cols teaching"><div class="card"><span class="micro">THE ORIGINAL DIET</span><p><b>${num(a.config.train_documents + a.config.validation_documents)} unique passages</b></p><p>Short classroom sentences about business, food, transport, and other familiar subjects.</p><div class="small-quote">“${esc(a.corpus_examples[0])}”</div><p class="hint">Actual saved training example.</p></div>
      <div class="card accent-card"><span class="micro">EXTRA LESSONS</span><p><b>+${num(b.manifest.new_unique_passages)} unique passages</b></p><p>Added practice with opposites, negation, spatial relations, and categories.</p><div class="small-quote">“${esc(b.manifest.files[0].preview.split('\n')[0])}”</div><p class="hint">Actual imported teaching text.</p></div></div>
      <div class="card"><span class="micro">TEACHING MATERIAL → PRACTICE → SAVED MODEL</span><p>Adding text is not an instant software update. Training repeatedly adjusts the model’s numbers using that text. Each experiment used <b>3,000 updates</b> and an initial learning rate of <b>0.001</b>.</p><p>90% of unique passages taught the model. The other 10% were held back to check predictions. They share sentence templates, so this is a narrow check.</p></div>
      <p class="hint">The 48 fixed tests are separate from training. Their prompts and answer keys were not teaching material. Ordinary facts may overlap; the original author chose additional exclusions.</p>${tryButton('customer')}
      ${evidence('expanded','corpus_manifest.json','Where the extra teaching text came from')}`;
  } else if (step === 1) {
    const v = i['embedding_' + vectorStage];
    content = `<p class="lesson-intro">A <b>token</b> is a word or punctuation unit. Its <b>ID</b> locates a row of numbers. That learned row is its <b>embedding</b>. The ID stays fixed within a run; the numbers change.</p>${modelPicker()}
      <div class="word-flow"><div><small>TOKEN</small><strong>${esc(i.token)}</strong></div><span>→</span><div><small>ROW / ID</small><strong>${i.token_id}</strong></div><span>→</span><div><small>EMBEDDING</small><strong>64 numbers</strong></div></div>
      <p class="hint">Customer is ID 28 in the starter vocabulary and ID 105 in the expanded vocabulary. IDs are row numbers, not meanings.</p>
      <div class="segmented" aria-label="Embedding stage"><button data-vector="before" aria-pressed="${vectorStage==='before'}">Before training</button><button data-vector="after" aria-pressed="${vectorStage==='after'}">After training</button></div>
      <div class="vectors" role="img" aria-label="64 embedding coordinates; green is positive, tan is negative. Exact values in the table below.">${v.map((x,n)=>`<span class="vector-cell ${x<0?'negative':x===0?'zero':''}" title="Coordinate ${n}: ${x}"></span>`).join('')}</div><p class="hint">Each square is one actual coordinate. Color shows its sign, not a named concept. No coordinate means “customer service.” Management is disappointed.</p>
      <details><summary>Show all 64 numbers, before and after</summary><div class="vector-table"><table><thead><tr><th>Coordinate</th><th>Before</th><th>After</th></tr></thead><tbody>${i.embedding_before.map((x,n)=>`<tr><td>${n}</td><td>${x.toFixed(7)}</td><td>${i.embedding_after[n].toFixed(7)}</td></tr>`).join('')}</tbody></table></div></details>
      <div class="card"><span class="micro">WHAT WORD COMES NEXT?</span><p>For the saved prefix <b>“${esc(i.prefix)}”</b>, these are the five most likely next tokens ${vectorStage==='before'?'before':'after'} training.</p><div class="prob-bars">${e.probabilities[vectorStage].map(([word,p])=>`<div class="prob-row"><span>${esc(word)}</span><progress max="1" value="${p}" aria-label="${esc(word)} ${pct(p)}"></progress><span>${pct(p)}</span></div>`).join('')}</div></div>
      ${tryButton('customer')}${evidence(lessonModel,'inspection.json','Original vectors and probabilities')}`;
  } else if (step === 2) {
    const update=i.first_update;
    content = `<p class="lesson-intro"><b>Guess → measure error → calculate a correction → update the numbers.</b> Repeat 3,000 times. This is training. A step updates a batch of 32 passages, not the whole corpus.</p>${modelPicker()}
      <div class="segmented" aria-label="Recorded training milestones">${['0','1500','3000'].map(s=>`<button data-milestone="${s}" aria-pressed="${sampleStep===s}">${num(s)} steps</button>`).join('')}</div>
      <div class="card"><span class="micro">${sampleStep==='0'?'BEFORE ONBOARDING':sampleStep==='1500'?'HALFWAY THROUGH ONBOARDING':'ELIGIBLE FOR A TITLE CHANGE'}</span><ol class="sample-list">${e.samples[sampleStep].map(s=>`<li>${s?esc(s):'[empty saved sample]'}</li>`).join('')}</ol><p class="hint">All saved samples at this milestone. Recorded evidence, not a live halfway model.</p></div>
      ${lossChart(e.history)}<div class="legend"><span><i class="train-key"></i>Training examples</span><span><i class="val-key"></i>Held-out examples</span></div>
      <p class="hint">Loss measures how much probability the model failed to assign to the actual next token. Lower is better on the same data. Lines connect just three measured points. Each panel has ${e.config.evaluation_panel_size.train} training / ${e.config.evaluation_panel_size.validation} held-out passages. Different corpora’s losses are not a ranking.</p>
      <details><summary>The complete measured loss table</summary><table><thead><tr><th>Steps</th><th>Training loss</th><th>Held-out loss</th></tr></thead><tbody>${e.history.map(r=>`<tr><td>${num(r.step)}</td><td>${r.training_loss.toFixed(4)}</td><td>${r.validation_loss.toFixed(4)}</td></tr>`).join('')}</tbody></table></details>
      <details><summary>One real gradient and weight update</summary><p>A gradient tells the optimizer how a small change would affect loss locally. AdamW uses gradients, running averages, and weight decay to update the neural network’s numbers.</p><table><tbody><tr><th>Recorded coordinate</th><td>${esc(update.token)} [${update.coordinate}]</td></tr><tr><th>Before</th><td>${update.before.toPrecision(8)}</td></tr><tr><th>Gradient</th><td>${update.gradient.toPrecision(8)}</td></tr><tr><th>Learning rate, first warmup step</th><td>${update.learning_rate}</td></tr><tr><th>After</th><td>${update.after.toPrecision(8)}</td></tr></tbody></table><p>The gradient was ${update.gradient>0?'positive':'negative'} and this coordinate moved ${update.after>update.before?'up':'down'}. AdamW is more than simply subtracting learning rate × gradient.</p></details>
      <div class="card accent-card"><span class="micro">WHAT WE CAN ACTUALLY SAY</span><p>Loss dropped and the saved text became more structured. These small panels cannot establish “no overfitting,” and three measurements cannot establish the lowest possible loss.</p></div>${tryButton('customer')}`;
  } else if (step === 3) {
    const evaluation = e.evaluations[evalStage], selected = evaluation.cases[selectedCase];
    content = `<p class="lesson-intro">Same 48 tests. Two different diets. A correct answer means the right word had the highest probability <b>among four choices</b>. Free text is generated separately.</p>
      <table><caption>Trained model comparison</caption><thead><tr><th>Measurement</th><th>Starter</th><th>Expanded</th></tr></thead><tbody>${[
        ['Correct / all tests','correct'],['Scorable / all tests','scorable'],['Accuracy on scorable tests','accuracy_scorable_cases']
      ].map(([label,key])=>`<tr><th>${label}</th>${['starter','expanded'].map(m=>`<td>${key.startsWith('accuracy')?pct(data.experiments[m].evaluations.final.summary.overall[key]):data.experiments[m].evaluations.final.summary.overall[key]+' / 48'}</td>`).join('')}</tr>`).join('')}</tbody></table>
      <div class="two-cols">${modelPicker()}<label>Evaluation stage<select id="eval-stage"><option value="final" ${evalStage==='final'?'selected':''}>Trained</option><option value="untrained" ${evalStage==='untrained'?'selected':''}>Untrained</option></select></label></div>
      <div class="legend"><span><i class="correct-key"></i>✓ Correct</span><span><i class="wrong-key"></i>× Incorrect / tied</span><span><i class="unknown-key"></i>– Unscorable</span></div>
      <div class="eval-grid" aria-label="All 48 evaluation cases">${evaluation.cases.map((r,n)=>{const state=r.score?'correct':['scored','tied'].includes(r.status)?'wrong':'unknown';return `<button class="eval-tile ${state}" data-case="${n}" aria-pressed="${selectedCase===n}" aria-label="Test ${n+1}, ${esc(r.category)}, ${state}">${n+1}<span aria-hidden="true">${r.score?'✓':state==='wrong'?'×':'–'}</span></button>`;}).join('')}</div>
      <div class="card case-detail" aria-live="polite"><span class="micro">TEST ${selectedCase+1} · ${esc(selected.category.replaceAll('_',' '))}</span><p class="small-quote">“${esc(selected.prompt)}”</p><p><b>Expected:</b> ${esc(selected.expected)} · <b>Model’s choice:</b> ${esc(selected.predicted_choice||'No scored choice')}</p><p><b>Actual free continuation:</b> ${esc(selected.generated_text||'[empty response]')}</p><p><b>Status:</b> ${esc(selected.status.replaceAll('_',' '))}</p>${selected.unknown_prompt_words.length||selected.unknown_choices.length?`<p>Missing prompt words: ${esc(selected.unknown_prompt_words.join(', ')||'none')}. Missing answer-choice words: ${esc(selected.unknown_choices.join(', ')||'none')}.</p>`:''}<details><summary>Four choices and recorded probabilities</summary><table><tbody>${selected.choices.map(c=>`<tr><th>${esc(c)}</th><td>${selected.choice_probabilities[c]===undefined?'Not scored':selected.choice_probabilities[c].toPrecision(5)}</td></tr>`).join('')}</tbody></table><p>${esc(selected.reason)}</p></details></div>
      <p class="hint">Any unknown prompt word or answer choice makes a case unscorable. It still counts as zero out of all 48. A higher score does not imply fluent or reliable chat.</p>
      <details><summary>All four result sets &amp; category breakdowns</summary>${allResults()}<table><thead><tr><th>Category</th><th>Correct</th><th>Scorable</th></tr></thead><tbody>${Object.entries(evaluation.summary.by_category).map(([k,v])=>`<tr><th>${esc(k.replaceAll('_',' '))}</th><td>${v.correct}/${v.total}</td><td>${v.scorable}/${v.total}</td></tr>`).join('')}</tbody></table><p>These are public development tests used to guide changes, not an unseen final benchmark.</p></details>
      <details><summary>Earlier expanded version (v1)</summary><p>v1 and v2 both scored 28/48 after training. v2 made 35 rather than 29 cases scorable. More usable vocabulary did not produce a higher total score.</p><a class="source-link" href="/evidence/llm_runs/${data.earlier_run}/language_evals/final/eval_results.json">Every v1 trained result ↗</a><br><a class="source-link" href="/evidence/llm_runs/${data.earlier_run}/language_evals/untrained/eval_results.json">Every v1 untrained result ↗</a></details>${tryButton('opposite','Investigate an actual failure →')}`;
  } else {
    const tokens=['<BOS>','the','customer'];
    content = `<p class="lesson-intro">It learned patterns. It also turned a cup into a goose. Both belong in the report.</p><div class="card accent-card"><span class="micro">THE GOOSE INCIDENT · RECORDED EXPANDED MODEL</span><p>Prompt: “${esc(examples.goose.prompt)}”</p><p class="small-quote">“${esc(examples.goose.expected)}”</p><p class="hint">Actual saved reply at temperature 0.8, seed ${examples.goose.seed}.</p>${tryButton('goose','Reproduce the goose incident →')}</div>
      <div class="card"><span class="micro">YOUR THREE-MINUTE PERFORMANCE REVIEW</span><p><b>1.</b> Try the goose example. Switch to untrained and compare.</p><p><b>2.</b> Keep the prompt and seed fixed. Try temperatures 0.3, 0.8, and 1.2. Replies may sometimes stay identical.</p><p><b>3.</b> Ask for homework help. Look at the unfamiliar words before judging the reply.</p>${tryButton('homework','Request academic assistance →')}</div>
      <details><summary>How attention uses earlier words</summary><p>Attention mixes information from earlier positions and the current position. It cannot read future tokens. This is one recorded head, not a complete explanation of a reply.</p>${modelPicker()}<div class="attention"><table><caption>Recorded attention for “the customer”</caption><thead><tr><th>Position</th>${tokens.map(t=>`<th>${esc(t)}</th>`).join('')}</tr></thead><tbody>${i.attention_rows.map((row,n)=>`<tr><th>${esc(tokens[n])}</th>${row.map(v=>`<td>${Number(v).toFixed(3)}</td>`).join('')}</tr>`).join('')}</tbody></table></div></details>
      <details><summary>How probabilities become generated text</summary><p>The model produces a score for each possible next token. Temperature scales those scores; softmax turns them into probabilities. A random draw selects a token, which is appended before the next prediction. An end marker or the 24-token limit stops the reply. No weights change.</p></details>
      <details><summary>Recorded temperature comparison</summary><p>These saved samples use the same starting token and sampling seed across temperatures. They do not retrain the model.</p>${[.3,.8,1.2].map(t=>`<p><b>Temperature ${t} · ${lessonModel} experiment</b></p><ol class="sample-list">${e.temperature_samples[String(t)].map(s=>`<li>${esc(s||'[empty response]')}</li>`).join('')}</ol>`).join('')}</details>
      <div class="card"><span class="micro">THE ACTUAL TAKEAWAY</span><p>Useful learning happened within a narrow set of sentence patterns. Fluent-looking text and a correct four-choice answer are not proof of general understanding.</p><p><b>Next experiment, not run:</b> repeat the expanded training with three seeds to see which results are stable. No new training is part of this lab.</p></div>`;
  }
  $('lesson-body').innerHTML = `<h3>${names[step]}</h3>${content}`;
  $('step-count').textContent = `0${step+1} / 05`;
  $('lesson-progress').textContent = `${step+1} of 5`;
  $('previous').disabled = step === 0;
  $('next').textContent = step===4 ? 'Try the raccoon →' : 'Next lesson →';
  document.querySelectorAll('[data-step]').forEach(b=>{if(Number(b.dataset.step)===step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
  if ($('lesson-model')) $('lesson-model').onchange = event => {lessonModel=event.target.value;lesson();};
  if ($('eval-stage')) $('eval-stage').onchange = event => {evalStage=event.target.value;lesson();};
  document.querySelectorAll('[data-vector]').forEach(b=>b.onclick=()=>{vectorStage=b.dataset.vector;lesson();});
  document.querySelectorAll('[data-milestone]').forEach(b=>b.onclick=()=>{sampleStep=b.dataset.milestone;lesson();});
  document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{selectedCase=Number(b.dataset.case);lesson();document.querySelector(`[data-case="${selectedCase}"]`).focus({preventScroll:true});});
  document.querySelectorAll('[data-try]').forEach(b=>b.onclick=()=>chooseExample(b.dataset.try,true));
  $('lesson-body').querySelectorAll('details').forEach((d,n)=>{d.open=!!openDetails[n];});
  if(focusSelector)document.querySelector(focusSelector)?.focus({preventScroll:true});
  renderedStep = step;
}

function allResults() {
  return `<table><thead><tr><th>Experiment / stage</th><th>Correct</th><th>Scorable</th><th>Coverage</th><th>Scorable accuracy</th></tr></thead><tbody>${['starter','expanded'].flatMap(m=>['untrained','final'].map(s=>{const a=data.experiments[m].evaluations[s].summary.overall;return `<tr><th><a href="/evidence/llm_runs/${data.experiments[m].run}/language_evals/${s}/eval_results.json">${m} / ${s==='final'?'trained':s}</a></th><td>${a.correct}/48</td><td>${a.scorable}/48</td><td>${pct(a.coverage)}</td><td>${pct(a.accuracy_scorable_cases)}</td></tr>`;})).join('')}</tbody></table>`;
}

function lossChart(history) {
  const y = v=>160-v/7*140, x = s=>43+s/3000*442;
  return `<svg class="chart" viewBox="0 0 530 205" role="img" aria-label="Training and held-out loss at steps 0, 1500, 3000; exact values below"><line x1="43" x2="485" y1="160" y2="160"/>${[0,2,4,6].map(v=>`<line x1="43" x2="485" y1="${y(v)}" y2="${y(v)}"/><text x="17" y="${y(v)+4}">${v}</text>`).join('')}<text x="5" y="13">Loss</text>${['training_loss','validation_loss'].map((k,n)=>`<polyline class="${n?'validation':'train'}" points="${history.map(r=>`${x(r.step)},${y(r[k])}`).join(' ')}"/>${history.map(r=>`<circle class="${n?'validation':'train'}" cx="${x(r.step)}" cy="${y(r[k])}" r="3"/>`).join('')}`).join('')}${history.map(r=>`<text x="${x(r.step)}" y="182" text-anchor="middle">${num(r.step)}</text>`).join('')}<text x="264" y="201" text-anchor="middle">Training steps · three measured milestones</text></svg>`;
}

function chooseExample(key, scroll=false) {
  const e=examples[key];
  $('prompt').value=e.prompt;$('seed').value=e.seed;$('model').value='expanded';$('stage').value='final';$('temperature').value='0.8';
  $('generation-status').textContent='Example loaded with its original settings. Press the button to generate a real reply.';
  if(scroll)$('playground').scrollIntoView({behavior:'smooth',block:'start'});
  $('prompt').focus({preventScroll:scroll});
}

function showReply(turn) {
  $('reply-area').hidden=false;
  $('reply').textContent=turn.response || '[empty response]';
  $('reply-meta').textContent=`${turn.experiment} · ${turn.stage==='final'?'trained':'untrained'}`;
  $('reply-context').textContent=`Generated for “${turn.prompt}” · temperature ${turn.temperature} · seed ${turn.seed}`;
  $('unknown').hidden=!turn.unknown_prompt_words.length && !turn.prompt_truncated;
  $('unknown').innerHTML=(turn.unknown_prompt_words.length?`<b>Unfamiliar words:</b> ${turn.unknown_prompt_words.map(w=>`<mark>${esc(w)}</mark>`).join(' ')}. These became unknown tokens.`:'')+(turn.prompt_truncated?'<p><b>Long prompt:</b> only its latest 48 tokens reached the first prediction.</p>':'');
  const match=Object.values(examples).find(e=>e.prompt===turn.prompt.trim() && e.expected===turn.response && e.seed===turn.seed && turn.experiment==='expanded' && turn.stage==='final' && turn.temperature===.8);
  $('management').hidden=!match;
  $('quip').textContent=match?.quip||'';
  $('why-text').textContent=match?.why||`This is a ${turn.stage==='final'?'trained':'randomly initialized'} model continuing your text, one token at a time. ${turn.unknown_prompt_words.length} distinct prompt words were unfamiliar. ${turn.prompt_truncated?'The prompt was shortened to the context limit.':'The prompt fit inside the context window.'} Temperature ${turn.temperature} controlled sampling randomness. These observations do not establish why it chose this particular wording.`;
  $('identity').textContent=JSON.stringify({run:turn.run,model_sha256:turn.model_sha256,completed_steps:turn.completed_steps,seed:turn.seed,temperature:turn.temperature,max_tokens:turn.max_tokens,fresh_context_per_prompt:true},null,2);
  $('turn-count').textContent=`${turns.length} interaction${turns.length===1?'':'s'}`;
  $('download').disabled=false;
  $('session-list').innerHTML=turns.map((t,n)=>`<div class="session-turn"><strong>${n+1}. ${esc(t.prompt)}</strong><code>${esc(t.experiment)} / ${esc(t.stage)} · T ${t.temperature} · seed ${t.seed}</code><p>${esc(t.response||'[empty response]')}</p></div>`).join('');
}

$('generate-form').onsubmit=async event=>{
  event.preventDefault();
  const payload={experiment:$('model').value,stage:$('stage').value,prompt:$('prompt').value,temperature:Number($('temperature').value),seed:Number($('seed').value)};
  if(!payload.prompt.trim()){ $('generation-status').textContent='Give the raccoon a sentence beginning first.';$('prompt').focus();return; }
  $('generate').disabled=true;$('generation-status').classList.remove('error');$('generation-status').textContent='The raccoon is consulting its actual model weights…';
  try {
    const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const turn=await response.json();if(!response.ok)throw new Error(turn.error||'Generation failed.');
    turns.push(turn);showReply(turn);$('generation-status').textContent='Generated locally. Model weights verified unchanged.';
  } catch(error){$('generation-status').textContent=error.message;$('generation-status').classList.add('error');}
  finally{$('generate').disabled=false;}
};
$('download').onclick=()=>{
  const blob=new Blob([JSON.stringify({format:'trashgpt-transcript-v1',exported_at:new Date().toISOString(),note:'Actual local model outputs. Management commentary is not model output.',turns},null,2)+'\n'],{type:'application/json'});
  const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`trashgpt-transcript-${new Date().toISOString().replaceAll(':','-')}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>chooseExample(b.dataset.example));
document.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{step=Number(b.dataset.step);lesson();});
$('previous').onclick=()=>{if(step>0){step--;lesson();}};
$('next').onclick=()=>{if(step<4){step++;lesson();}else{$('playground').scrollIntoView({behavior:'smooth'});$('prompt').focus({preventScroll:true});}};
fetch('/api/experiments').then(async response=>{if(!response.ok)throw new Error('Could not load the saved experiments. Check the local server and evidence files.');return response.json();}).then(result=>{
  data=result;Object.entries(data.curated_examples).forEach(([key,value])=>Object.assign(examples[key],value));$('prompt').value=examples.goose.prompt;$('seed').value=examples.goose.seed;$('starter-score').textContent=`${data.experiments.starter.evaluations.final.summary.overall.correct} / 48`;$('expanded-score').textContent=`${data.experiments.expanded.evaluations.final.summary.overall.correct} / 48`;
  $('loading').hidden=true;$('app').hidden=false;lesson();
}).catch(error=>{$('loading').textContent=error.message;$('loading').classList.add('error');});
