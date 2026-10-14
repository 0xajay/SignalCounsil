import assert from 'node:assert'
import { 
  cleanSymbol,
  formatSymbolForMarket, 
  parseTimeframe, 
  detectActiveChart 
} from './chartDetection.ts'

console.log('--- Testing cleanSymbol & formatSymbolForMarket across Global Markets ---')

// 1. Clean symbols across global markets (no .NS, no .BO, no .L, no .TO, etc.)
assert.strictEqual(cleanSymbol('RELIANCE'), 'RELIANCE')
assert.strictEqual(cleanSymbol('RELIANCE.NS'), 'RELIANCE')
assert.strictEqual(cleanSymbol('NSE:RELIANCE'), 'RELIANCE')
assert.strictEqual(cleanSymbol('TCS.BO'), 'TCS')
assert.strictEqual(cleanSymbol('BSE:TCS'), 'TCS')
assert.strictEqual(cleanSymbol('LSE:VOD'), 'VOD')
assert.strictEqual(cleanSymbol('VOD.L'), 'VOD')
assert.strictEqual(cleanSymbol('TSX:SHOP'), 'SHOP')
assert.strictEqual(cleanSymbol('SHOP.TO'), 'SHOP')
assert.strictEqual(cleanSymbol('ASX:BHP'), 'BHP')
assert.strictEqual(cleanSymbol('BHP.AX'), 'BHP')
assert.strictEqual(cleanSymbol('XETRA:SAP'), 'SAP')
assert.strictEqual(cleanSymbol('SAP.DE'), 'SAP')
assert.strictEqual(cleanSymbol('BINANCE:BTCUSDT'), 'BTCUSDT')
assert.strictEqual(cleanSymbol('FX:EURUSD'), 'EURUSD')
assert.strictEqual(cleanSymbol('^NSEI'), 'NIFTY')
assert.strictEqual(cleanSymbol('^BSESN'), 'SENSEX')
assert.strictEqual(cleanSymbol('^GSPC'), 'SPX')
assert.strictEqual(cleanSymbol('^IXIC'), 'NDX')
assert.strictEqual(cleanSymbol('NASDAQ:AAPL'), 'AAPL')
assert.strictEqual(cleanSymbol('AAPL'), 'AAPL')

assert.strictEqual(formatSymbolForMarket('RELIANCE', 'NSE'), 'RELIANCE')
assert.strictEqual(formatSymbolForMarket('VOD.L', 'LSE'), 'VOD')
assert.strictEqual(formatSymbolForMarket('AAPL', 'NASDAQ'), 'AAPL')
console.log('✓ cleanSymbol and formatSymbolForMarket passed for all global markets')

console.log('\n--- Testing parseTimeframe ---')
assert.strictEqual(parseTimeframe('15m'), '15m')
assert.strictEqual(parseTimeframe('15'), '15m')
assert.strictEqual(parseTimeframe('5m'), '15m')
assert.strictEqual(parseTimeframe('30m'), '15m')
assert.strictEqual(parseTimeframe('60'), '1h')
assert.strictEqual(parseTimeframe('1h'), '1h')
assert.strictEqual(parseTimeframe('4h'), '1h')
assert.strictEqual(parseTimeframe('D'), '1d')
assert.strictEqual(parseTimeframe('1D'), '1d')
assert.strictEqual(parseTimeframe('1d'), '1d')
assert.strictEqual(parseTimeframe('daily'), '1d')
assert.strictEqual(parseTimeframe('W'), '1wk')
assert.strictEqual(parseTimeframe('1W'), '1wk')
assert.strictEqual(parseTimeframe('weekly'), '1wk')
assert.strictEqual(parseTimeframe(null), null)
console.log('✓ Timeframe parsing passed')

console.log('\n--- Testing detectActiveChart (Global In-Chart DOM & Live Title) ---')

// Mock chrome environment
const mockGlobalChrome = (tab: { url: string; title?: string; id?: number }, domResult?: any) => {
  (globalThis as any).chrome = {
    tabs: {
      query: (_queryInfo: any, callback: (tabs: any[]) => void) => {
        callback([{ ...tab, id: tab.id || 101 }])
      }
    },
    scripting: {
      executeScript: async () => {
        if (domResult) {
          return [{ result: domResult }]
        }
        return []
      }
    }
  }
}

async function runChartTests() {
  // Test 1: TradingView Chart with STATIC URL (/chart/abc12345/) and LIVE Indian Stock
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/abc12345/',
    title: 'RELIANCE 2,980.50 INR ▲ +12.30 (+0.41%) — TradingView'
  })
  const tvNse = await detectActiveChart()
  assert.ok(tvNse)
  assert.strictEqual(tvNse.market, 'NSE')
  assert.strictEqual(tvNse.rawSymbol, 'RELIANCE')
  assert.strictEqual(tvNse.formattedSymbol, 'RELIANCE')
  console.log('✓ TradingView Indian equity detection passed')

  // Test 2: TradingView UK Stock (LSE:VOD)
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/layout_lse/',
    title: 'Vodafone Group Plc — LSE:VOD — 1h — TradingView'
  })
  const tvLse = await detectActiveChart()
  assert.ok(tvLse)
  assert.strictEqual(tvLse.market, 'LSE')
  assert.strictEqual(tvLse.rawSymbol, 'VOD')
  assert.strictEqual(tvLse.formattedSymbol, 'VOD')
  assert.strictEqual(tvLse.timeframe, '1h')
  console.log('✓ TradingView UK (LSE) equity detection passed')

  // Test 3: TradingView Canadian Stock (TSX:SHOP)
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/layout_tsx/',
    title: 'Shopify Inc. SHOP 115.40 CAD ▲ +2.10 (+1.85%) — TradingView'
  })
  const tvTsx = await detectActiveChart()
  assert.ok(tvTsx)
  assert.strictEqual(tvTsx.market, 'TSX')
  assert.strictEqual(tvTsx.rawSymbol, 'SHOP')
  assert.strictEqual(tvTsx.formattedSymbol, 'SHOP')
  console.log('✓ TradingView Canada (TSX) equity detection passed')

  // Test 4: TradingView Australian Stock (ASX:BHP)
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/layout_asx/',
    title: 'BHP Group Ltd BHP 42.50 AUD — 15m — TradingView'
  })
  const tvAsx = await detectActiveChart()
  assert.ok(tvAsx)
  assert.strictEqual(tvAsx.market, 'ASX')
  assert.strictEqual(tvAsx.rawSymbol, 'BHP')
  assert.strictEqual(tvAsx.formattedSymbol, 'BHP')
  assert.strictEqual(tvAsx.timeframe, '15m')
  console.log('✓ TradingView Australia (ASX) equity detection passed')

  // Test 5: TradingView Crypto (BINANCE:BTCUSDT)
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/crypto_layout/',
    title: 'Bitcoin / TetherUS — BINANCE:BTCUSDT — 15m — TradingView'
  })
  const tvCrypto = await detectActiveChart()
  assert.ok(tvCrypto)
  assert.strictEqual(tvCrypto.market, 'BINANCE')
  assert.strictEqual(tvCrypto.rawSymbol, 'BTCUSDT')
  assert.strictEqual(tvCrypto.formattedSymbol, 'BTCUSDT')
  assert.strictEqual(tvCrypto.timeframe, '15m')
  console.log('✓ TradingView Crypto (BINANCE) detection passed')

  // Test 6: TradingView Forex (FX:EURUSD)
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/forex_layout/',
    title: 'Euro / US Dollar — FX:EURUSD — 1h — TradingView'
  })
  const tvForex = await detectActiveChart()
  assert.ok(tvForex)
  assert.strictEqual(tvForex.market, 'FX')
  assert.strictEqual(tvForex.rawSymbol, 'EURUSD')
  assert.strictEqual(tvForex.formattedSymbol, 'EURUSD')
  assert.strictEqual(tvForex.timeframe, '1h')
  console.log('✓ TradingView Forex (FX) detection passed')

  // Test 7: TradingView in-chart DOM overrides stale URL
  mockGlobalChrome({
    url: 'https://www.tradingview.com/chart/layout99/?symbol=OLD_STALE_TICKER',
    title: 'TradingView Chart'
  }, {
    symbolBtnText: 'SAP',
    legendTitleText: 'SAP',
    legendText: 'SAP · 1h · XETRA',
    intervalBtnText: '1h',
    docTitle: 'SAP 180.50 EUR — TradingView'
  })
  const tvDom = await detectActiveChart()
  assert.ok(tvDom)
  assert.strictEqual(tvDom.market, 'XETRA')
  assert.strictEqual(tvDom.rawSymbol, 'SAP')
  assert.strictEqual(tvDom.formattedSymbol, 'SAP')
  assert.strictEqual(tvDom.timeframe, '1h')
  console.log('✓ TradingView DOM overrides stale URL for German XETRA passed')

  console.log('\nALL GLOBAL MARKET UNIT TESTS PASSED!')
}

runChartTests()
