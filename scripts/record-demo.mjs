import { spawn } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import puppeteer from 'puppeteer-core';

const chrome = [process.env.CHROME_PATH, '/usr/local/bin/google-chrome', '/usr/bin/google-chrome'].find(
  (path) => path && existsSync(path),
);
if (!chrome) {
  console.error('Chrome not found. Set CHROME_PATH.');
  process.exit(1);
}
if (!existsSync('dist/index.html')) {
  console.error('Run npm run build first.');
  process.exit(1);
}

const port = 4191;
const outDir = 'docs/demo-frames';
mkdirSync(outDir, { recursive: true });
const server = spawn(
  'npx',
  ['vite', 'preview', '--host', '127.0.0.1', '--port', String(port), '--strictPort'],
  { stdio: 'ignore' },
);

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForServer() {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/`)).ok) return;
    } catch {
      /* starting */
    }
    await delay(250);
  }
  throw new Error('preview did not start');
}

const browser = await puppeteer.launch({
  executablePath: chrome,
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream'],
});

try {
  await waitForServer();
  const page = await browser.newPage();
  await page.setViewport({ width: 1100, height: 780, deviceScaleFactor: 1 });
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => document.body.innerText.includes('English glosses'));
  await page.evaluate(() => {
    const larger = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'A++');
    larger?.click();
  });
  const clip = { x: 0, y: 0, width: 1100, height: 700 };
  const shot = async (name) => {
    await delay(350);
    await page.screenshot({ path: join(outDir, name), clip });
  };
  await shot('01-ready.png');
  for (const name of ['02-hello.png', '03-thank-you.png', '04-please.png']) {
    await page.click('#replay-held-out');
    await shot(name);
  }
  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'Undo');
    button?.click();
  });
  await shot('05-undo.png');
  await page.type('#reply-input', 'good morning');
  await page.evaluate(() => {
    const button = [...document.querySelectorAll('button')].find((node) => node.textContent?.trim() === 'Show text');
    button?.click();
  });
  await page.waitForFunction(() => document.querySelector('.hearing-hero')?.textContent?.includes('good morning'));
  await shot('06-reply.png');
  console.log('frames written');
} finally {
  server.kill('SIGTERM');
  browser.close().catch(() => {});
  setTimeout(() => process.exit(0), 400);
}
