const puppeteer = require("puppeteer");
const path = require("path");

const SHOTS = [
  { name: "j20_hero", shot: "j20_hero", w: 1600, h: 900 },
  { name: "j20_canopy", shot: "j20_canopy", w: 1600, h: 900 },
  { name: "j20_rear", shot: "j20_rear", w: 1600, h: 900 },
  { name: "j20_detail", shot: "j20_detail", w: 1600, h: 900 },
  { name: "f16_hero", shot: "f16_hero", w: 1600, h: 900 },
  { name: "f16_canopy", shot: "f16_canopy", w: 1600, h: 900 },
  { name: "f15_hero", shot: "f15_hero", w: 1600, h: 900 },
  { name: "f15_canopy", shot: "f15_canopy", w: 1600, h: 900 },
  { name: "su57_hero", shot: "su57_hero", w: 1600, h: 900 },
  { name: "su57_canopy", shot: "su57_canopy", w: 1600, h: 900 },
  { name: "missiles", shot: "missiles", w: 1600, h: 900 },
];

const BASE = process.env.BASE || "http://127.0.0.1:8099/render_demo/src/index.html";

(async () => {
  const browser = await puppeteer.launch({
    headless: true,
    executablePath: process.env.CHROME
      || "/root/.cache/puppeteer/chrome/linux-155.0.8059.39/chrome-linux64/chrome",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--use-gl=angle",
      "--use-angle=swiftshader",
      "--enable-unsafe-swiftshader",
      "--disable-dev-shm-usage",
      "--force-device-scale-factor=1",
      "--hide-scrollbars",
    ],
  });
  const only = process.argv[2];
  for (const s of SHOTS) {
    if (only && s.name !== only) continue;
    const page = await browser.newPage();
    page.on("console", (m) => {
      const t = m.text();
      if (/error|Error|fail|WebGL/i.test(t)) console.log(`[${s.name}] console: ${t}`);
    });
    page.on("pageerror", (e) => console.log(`[${s.name}] pageerror: ${e.message}`));
    const w = +(process.env.W || s.w);
    const h = +(process.env.H || s.h);
    await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 });
    const url = `${BASE}?shot=${s.shot}&w=${w}&h=${h}`
      + (process.env.DBG ? `&dbg=${process.env.DBG}` : "");
    const t0 = Date.now();
    await page.goto(url, { waitUntil: "load", timeout: 120000 });
    try {
      await page.waitForFunction("window.__shotReady === true || window.__shotError", { timeout: 240000 });
    } catch (e) {
      console.log(`[${s.name}] 超时`);
    }
    const err = await page.evaluate(() => window.__shotError || null);
    if (err) {
      console.log(`[${s.name}] 渲染错误: ${err}`);
      await page.close();
      continue;
    }
    const out = path.join(__dirname, "out", `${process.env.W ? "preview_" : ""}${s.name}.png`);
    await page.screenshot({ path: out });
    console.log(`[${s.name}] ${out}  ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    await page.close();
  }
  await browser.close();
})();
