import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import puppeteer from 'puppeteer-core';

const require = createRequire(import.meta.url);
const axeSource = readFileSync(require.resolve('axe-core/axe.min.js'), 'utf8');

const chromeCandidates = [
  process.env.CHROME_PATH,
  '/usr/local/bin/google-chrome',
  '/usr/bin/google-chrome',
  '/usr/bin/google-chrome-stable',
].filter(Boolean);
const chrome = chromeCandidates.find((path) => existsSync(path));
if (!chrome) {
  console.error('Chrome not found. Set CHROME_PATH.');
  process.exit(1);
}
if (!existsSync('dist/sw.js') || !existsSync('dist/index.html')) {
  console.error('dist/sw.js is missing. Run npm run build first.');
  process.exit(1);
}

const port = Number(process.env.E2E_PORT || 4311);
const server = spawn(
  'npx',
  ['vite', 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  { detached: true, stdio: ['ignore', 'pipe', 'pipe'] },
);
let serverLog = '';
let serverDied = false;
server.on('exit', () => {
  serverDied = true;
});
server.stdout.on('data', (chunk) => {
  serverLog += chunk.toString();
});
server.stderr.on('data', (chunk) => {
  serverLog += chunk.toString();
});

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForServer() {
  const url = `http://127.0.0.1:${port}/`;
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return url;
    } catch {
      /* preview is still starting */
    }
    if (serverDied || serverLog.includes('already in use')) break;
    await delay(250);
  }
  throw new Error(`Preview did not start.\n${serverLog}`);
}

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function expectNoSeriousAxeViolations(page, label) {
  await page.addScriptTag({ content: axeSource });
  const violations = await page.evaluate(async () => {
    const outcome = await window.axe.run(document, { resultTypes: ['violations'] });
    return outcome.violations
      .filter((item) => item.impact === 'serious' || item.impact === 'critical')
      .map((item) => {
        const where = item.nodes.slice(0, 4).map((node) => `${node.target.join(' ')} :: ${(node.html || '').slice(0, 140)}`).join(' | ');
        return `${item.id} (${item.nodes.length}): ${where}`;
      });
  });
  assert(violations.length === 0, `axe ${label}: ${violations.join('; ')}`);
}

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: [
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--use-fake-ui-for-media-stream',
    '--use-fake-device-for-media-stream',
  ],
});

try {
  const url = await waitForServer();
  const page = await browser.newPage();
  page.setDefaultTimeout(45000);
  await page.goto(url, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => document.body.innerText.includes('English glosses'));
  const speechText = await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.includes('Speaking aloud'));
    return button?.textContent ?? '';
  });
  assert(speechText.includes('off'), `Speech should start off, saw "${speechText}"`);
  await expectNoSeriousAxeViolations(page, 'light');

  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'Contrast');
    button?.click();
  });
  await page.waitForSelector('[data-theme="contrast"]');
  await expectNoSeriousAxeViolations(page, 'contrast');

  await page.click('#replay-held-out');
  await page.waitForSelector('.caption-word');
  let word = await page.$eval('.caption-word', (node) => node.textContent?.trim() ?? '');
  assert(word.length > 0, 'Replay did not add an English word');
  const english = await page.$eval('#english-line', (node) => node.textContent?.trim() ?? '');
  assert(english === 'Hello.', `Expected an English sentence "Hello.", saw "${english}"`);
  await page.click('#replay-held-out');
  await page.waitForFunction(() => document.querySelectorAll('.caption-word').length >= 2);
  const sentence = await page.$eval('#english-line', (node) => node.textContent?.trim() ?? '');
  assert(/hello/i.test(sentence) && /thank you/i.test(sentence), `Expected two replayed words, saw "${sentence}"`);
  await page.evaluate(() => {
    const clear = [...document.querySelectorAll('.caption-stage .caption-actions button')].find((node) => node.textContent?.trim() === 'Clear');
    if (!(clear instanceof HTMLButtonElement)) throw new Error('caption Clear is missing');
    clear.click();
  });
  await page.waitForFunction(() => document.querySelectorAll('.caption-word').length === 0);
  await page.click('#replay-held-out');
  await page.waitForSelector('.caption-word');
  word = await page.$eval('.caption-word', (node) => node.textContent?.trim() ?? '');
  assert(word.length > 0, 'Replay after Clear did not add a word');
  await page.type('#personal-label', 'hello');
  await page.click('#save-personal');
  await page.waitForFunction(() => document.querySelector('#personal-message')?.textContent?.includes('Saved'));
  const flash = await page.$eval('.commit-flash', (node) => node.textContent ?? '');
  assert(flash.includes(word), `Commit flash did not name ${word}`);

  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'Undo');
    button?.click();
  });
  await page.waitForFunction(() => document.querySelectorAll('.caption-word').length === 0);

  await page.type('#reply-input', 'good morning');
  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'Show text');
    button?.click();
  });
  await page.waitForFunction(() => document.querySelector('.hearing-hero')?.textContent?.includes('good morning'));
  await page.waitForFunction(() => [...document.querySelectorAll('.chat-item-hearing p')].some((node) => node.textContent?.includes('good morning')));

  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  await page.reload({ waitUntil: 'networkidle0' });
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null);
  await page.setOfflineMode(true);
  await page.reload({ waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => document.body.innerText.includes('English glosses'));
  await page.click('#replay-held-out');
  await page.waitForSelector('.caption-word');
  const offlineWord = await page.$eval('.caption-word', (node) => node.textContent?.trim() ?? '');
  assert(offlineWord.length > 0, 'Offline replay did not add a word');
  console.log(`e2e ok: online "${word}", offline "${offlineWord}"`);
} catch (error) {
  console.error(error);
  console.error(serverLog);
  process.exitCode = 1;
} finally {
  try {
    process.kill(-server.pid, 'SIGTERM');
  } catch {
    server.kill('SIGTERM');
  }
  browser.close().catch(() => {});
  setTimeout(() => process.exit(process.exitCode ?? 0), 500);
}
