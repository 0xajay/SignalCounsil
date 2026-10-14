export type MarketType = string

export interface DetectedChartInfo {
  market: MarketType | null
  rawSymbol: string
  formattedSymbol: string
  timeframe: '15m' | '1h' | '1d' | '1wk' | null
  source: string
  pageTitle?: string
}

export function cleanSymbol(rawSymbol: string): string {
  let sym = rawSymbol.trim().toUpperCase()
  if (!sym) return ''

  // Strip prefixes like NSE: or BSE: or NASDAQ: or NYSE: or LSE: or BINANCE: etc.
  if (sym.includes(':')) {
    sym = sym.split(':')[1].trim()
  }

  // Strip common global exchange suffixes (.NS, .BO, .L, .TO, .AX, .DE, .PA, .HK, .T)
  sym = sym.replace(/\.(NS|BO|L|TO|AX|DE|PA|HK|T|MI|MC|SW|SA|MX|JO)$/i, '')

  // Map provider indexes back to clean friendly names if detected
  if (sym === '^NSEI') return 'NIFTY'
  if (sym === '^NSEBANK') return 'BANKNIFTY'
  if (sym === '^BSESN') return 'SENSEX'
  if (sym === '^GSPC') return 'SPX'
  if (sym === '^IXIC') return 'NDX'
  if (sym === '^DJI') return 'DJI'
  if (sym === '^FTSE') return 'FTSE'
  if (sym === '^GDAXI') return 'DAX'

  return sym
}

export function formatSymbolForMarket(
  rawSymbol: string, 
  _targetMarket?: MarketType | null
): string {
  return cleanSymbol(rawSymbol)
}

export function parseTimeframe(rawInterval: string | null | undefined): '15m' | '1h' | '1d' | '1wk' | null {
  if (!rawInterval) return null
  const s = rawInterval.toLowerCase().trim()
  if (['15', '15m', '15min', '5', '5m', '5min', '30', '30m', '30min'].includes(s)) return '15m'
  if (['60', '60m', '1h', '1hr', '2h', '4h', '120', '240'].includes(s)) return '1h'
  if (['d', '1d', 'day', 'daily', '1day'].includes(s)) return '1d'
  if (['w', '1w', 'wk', '1wk', 'week', 'weekly', '1week', 'm', '1mo', 'month', 'monthly'].includes(s)) return '1wk'
  return null
}

export async function detectActiveChart(): Promise<DetectedChartInfo | null> {
  if (typeof chrome === 'undefined' || !chrome.tabs || !chrome.tabs.query) {
    return null
  }

  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const currentTab = tabs[0]
      if (!currentTab || !currentTab.url) {
        resolve(null)
        return
      }

      const urlStr = currentTab.url
      const titleStr = currentTab.title || ''

      let detectedMarket: MarketType | null = null
      let rawSymbol = ''
      let detectedTf: '15m' | '1h' | '1d' | '1wk' | null = null
      let source = 'Active Tab'

      // 1. TradingView detection (Works for all global markets)
      // On TradingView chart pages (/chart/*), the app is an interactive SPA.
      // Changing the stock in the chart DOES NOT update URL query parameters.
      // Therefore, we NEVER rely on URL query params for TradingView charts.
      // We extract directly from the live DOM and live document.title.
      if (urlStr.includes('tradingview.com')) {
        source = 'TradingView'

        // A. Primary: Extract live state from DOM via chrome.scripting
        if (typeof chrome.scripting !== 'undefined' && currentTab.id) {
          try {
            const results = await chrome.scripting.executeScript({
              target: { tabId: currentTab.id },
              func: () => {
                const symbolBtn = document.querySelector('#header-toolbar-symbol-search') ||
                                  document.querySelector('[id^="header-toolbar-symbol-search"]') ||
                                  document.querySelector('button[id*="symbol"]')
                
                const legendTitle = document.querySelector('[data-name="legend-source-title"]') ||
                                    document.querySelector('[data-name="legend-series-item"] [class*="title-"]') ||
                                    document.querySelector('[data-name="legend-series-item"] [class*="seriesTitle-"]')
                const legendExchange = document.querySelector('[data-name="legend-series-item"] [class*="exchange-"]') ||
                                       document.querySelector('[data-name="legend-source-title"] + [class*="exchange"]') ||
                                       document.querySelector('[class*="pane-legend-line"] [class*="exchange-"]')
                const legendEl = document.querySelector('[data-name="legend-series-item"]') ||
                                 document.querySelector('.pane-legend-line') ||
                                 document.querySelector('div[class*="titlesWrapper"]')
                
                const intervalBtn = document.querySelector('#header-toolbar-intervals') ||
                                    document.querySelector('[id^="header-toolbar-intervals"]') ||
                                    document.querySelector('button[data-name="time-interval-button"]') ||
                                    document.querySelector('button[data-name="timeframe-group"]')

                const fullSymbolAttr = document.querySelector('[data-symbol-full]')?.getAttribute('data-symbol-full') ||
                                       document.querySelector('[data-symbol]')?.getAttribute('data-symbol') || ''

                return {
                  symbolBtnText: symbolBtn?.textContent?.trim() || '',
                  legendTitleText: legendTitle?.textContent?.trim() || '',
                  legendExchangeText: legendExchange?.textContent?.trim() || '',
                  legendText: legendEl?.textContent || '',
                  intervalBtnText: intervalBtn?.textContent?.trim() || '',
                  fullSymbolAttr: fullSymbolAttr.trim(),
                  docTitle: document.title || ''
                }
              }
            })

            if (results && results[0] && results[0].result) {
              const domData = results[0].result
              
              if (domData.intervalBtnText) {
                detectedTf = parseTimeframe(domData.intervalBtnText)
              }
              
              if (domData.fullSymbolAttr && domData.fullSymbolAttr.includes(':')) {
                const [ex, s] = domData.fullSymbolAttr.split(':')
                detectedMarket = ex.trim().toUpperCase()
                rawSymbol = cleanSymbol(s.trim())
              }

              const candidate = domData.legendTitleText || domData.symbolBtnText
              if (candidate && !rawSymbol) {
                const token = candidate.split(/[\s·,]/)[0].trim()
                if (token && token.length < 20 && !['CHART', 'TRADINGVIEW', 'COMPARE', 'INDICATORS'].includes(token.toUpperCase())) {
                  rawSymbol = token
                }
              }

              if (domData.legendExchangeText) {
                detectedMarket = domData.legendExchangeText.trim().toUpperCase()
              } else if (domData.legendText && !detectedMarket) {
                // Check all known global markets in legend
                const exMatch = domData.legendText.match(/\b(NSE|BSE|NASDAQ|NYSE|AMEX|LSE|TSX|TSXV|ASX|HKEX|TSE|XETRA|EURONEXT|BINANCE|COINBASE|BYBIT|FX|FOREX|MCX)\b/i)
                if (exMatch) {
                  detectedMarket = exMatch[1].toUpperCase()
                }

                if (!detectedTf) {
                  const tfMatch = domData.legendText.match(/·\s*(15m|1h|60m|1D|D|1W|W)\s*·/i)
                  if (tfMatch) detectedTf = parseTimeframe(tfMatch[1])
                }
              }
            }
          } catch {
            // Scripting fallback
          }
        }

        // Regional TradingView domain inference (e.g. in.tradingview.com -> NSE, uk.tradingview.com -> LSE)
        if (!detectedMarket) {
          if (urlStr.includes('in.tradingview.com')) detectedMarket = 'NSE'
          else if (urlStr.includes('uk.tradingview.com')) detectedMarket = 'LSE'
          else if (urlStr.includes('ca.tradingview.com')) detectedMarket = 'TSX'
          else if (urlStr.includes('au.tradingview.com')) detectedMarket = 'ASX'
          else if (urlStr.includes('de.tradingview.com')) detectedMarket = 'XETRA'
          else if (urlStr.includes('fr.tradingview.com')) detectedMarket = 'EURONEXT'
          else if (urlStr.includes('jp.tradingview.com')) detectedMarket = 'TSE'
        }

        // B. Authoritative Fallback: Live document.title / tab.title
        if (titleStr) {
          // Check for exchange prefix in title: e.g. LSE:VOD, TSX:SHOP, NSE:RELIANCE, NASDAQ:AAPL
          const exMatch = titleStr.match(/\b([A-Z0-9_-]+):([A-Z0-9_.-]+)\b/i)
          if (exMatch) {
            const ex = exMatch[1].toUpperCase()
            detectedMarket = ex === 'NYSE' ? 'NASDAQ' : ex
            if (!rawSymbol) rawSymbol = exMatch[2]
          }

          // Check for TradingView price & currency pattern:
          // e.g. "RELIANCE 2,980.50 INR", "AAPL 224.50 USD", "VOD 75.20 GBp", "SAP 180.50 EUR", "BHP 42.10 AUD", "Shopify Inc. SHOP 115.40 CAD"
          const priceTitleMatch = titleStr.match(/(?:^|[\s(])([A-Z0-9_\-.]+)(?:\s+50)?\s+[\d,.]+\s+([A-Za-z]{3,4})/i)
          if (priceTitleMatch) {
            if (!rawSymbol) rawSymbol = priceTitleMatch[1]
            const curr = priceTitleMatch[2].toUpperCase()
            if (!detectedMarket) {
              if (curr === 'INR' || curr === 'RS') detectedMarket = /\bBSE\b/i.test(titleStr) ? 'BSE' : 'NSE'
              else if (curr === 'USD') detectedMarket = 'NASDAQ'
              else if (curr === 'GBP') detectedMarket = 'LSE'
              else if (curr === 'CAD') detectedMarket = 'TSX'
              else if (curr === 'AUD') detectedMarket = 'ASX'
              else if (curr === 'EUR') detectedMarket = 'XETRA'
              else if (curr === 'JPY') detectedMarket = 'TSE'
              else if (curr === 'HKD') detectedMarket = 'HKEX'
              else if (curr === 'USDT') detectedMarket = 'BINANCE'
            }
          }

          // Check for currency symbols or indicators in title (₹, INR, Rs -> Indian NSE)
          if (!detectedMarket) {
            if (titleStr.includes('₹') || /\b(INR|Rs)\b/i.test(titleStr)) {
              detectedMarket = /\bBSE\b/i.test(titleStr) ? 'BSE' : 'NSE'
            }
          }

          // Global exchange word in title
          if (!detectedMarket) {
            const globalExMatch = titleStr.match(/\b(NSE|BSE|NASDAQ|NYSE|AMEX|LSE|TSX|TSXV|ASX|HKEX|TSE|XETRA|EURONEXT|BINANCE|COINBASE|BYBIT|FX|FOREX|MCX)\b/i)
            if (globalExMatch) {
              detectedMarket = globalExMatch[1].toUpperCase()
            }
          }

          // Fallback leading symbol match
          if (!rawSymbol) {
            const titleSymMatch = titleStr.match(/^([A-Z0-9_\-.]+)\s+/i)
            if (titleSymMatch && titleSymMatch[1] && !['TRADINGVIEW', 'CHART', 'FREE'].includes(titleSymMatch[1].toUpperCase())) {
              rawSymbol = titleSymMatch[1]
            }
          }

          // Fallback timeframe from title
          if (!detectedTf) {
            const tfMatch = titleStr.match(/\b(15m|15min|1h|60m|1d|daily|1w|1wk|weekly|1D|1W)\b/i)
            if (tfMatch) detectedTf = parseTimeframe(tfMatch[1])
          }
        }

        // C. Path pattern ONLY for dedicated /symbols/ landing pages (NOT /chart/)
        if (!rawSymbol && urlStr.includes('/symbols/')) {
          const symMatch = urlStr.match(/symbols\/([A-Za-z0-9_\-.]+)/)
          if (symMatch && symMatch[1]) {
            const pathSym = symMatch[1]
            if (pathSym.includes('-')) {
              const [ex, s] = pathSym.split('-')
              detectedMarket = ex.toUpperCase()
              rawSymbol = s
            } else {
              rawSymbol = pathSym
            }
          }
        }
      }

      // 2. Google Finance quote detection (dedicated URL path)
      else if (urlStr.includes('google.com/finance/quote/')) {
        source = 'Google Finance'
        const gfMatch = urlStr.match(/quote\/([A-Z0-9_\-.]+):([A-Z]+)/i)
        if (gfMatch) {
          rawSymbol = gfMatch[1]
          detectedMarket = gfMatch[2].toUpperCase()
        }
      }

      // 3. Yahoo Finance quote detection (dedicated URL path)
      else if (urlStr.includes('finance.yahoo.com/quote/')) {
        source = 'Yahoo Finance'
        const yfMatch = urlStr.match(/quote\/([A-Z0-9_\-.^]+)/i)
        if (yfMatch) {
          const yfSym = yfMatch[1].toUpperCase()
          if (yfSym.includes('.')) {
            const parts = yfSym.split('.')
            rawSymbol = cleanSymbol(parts[0])
            const sfx = `.${parts[1]}`
            if (sfx === '.NS') detectedMarket = 'NSE'
            else if (sfx === '.BO') detectedMarket = 'BSE'
            else if (sfx === '.L') detectedMarket = 'LSE'
            else if (sfx === '.TO') detectedMarket = 'TSX'
            else if (sfx === '.AX') detectedMarket = 'ASX'
            else if (sfx === '.DE') detectedMarket = 'XETRA'
            else if (sfx === '.PA') detectedMarket = 'EURONEXT'
            else if (sfx === '.T') detectedMarket = 'TSE'
            else if (sfx === '.HK') detectedMarket = 'HKEX'
          } else if (yfSym.endsWith('-USD')) {
            rawSymbol = yfSym.replace('-USD', '')
            detectedMarket = 'CRYPTO'
          } else {
            rawSymbol = cleanSymbol(yfSym)
            detectedMarket = 'NASDAQ'
          }
        }
      }

      if (!rawSymbol) {
        resolve(null)
        return
      }

      const cleaned = cleanSymbol(rawSymbol)
      resolve({
        market: detectedMarket,
        rawSymbol: cleaned,
        formattedSymbol: cleaned,
        timeframe: detectedTf,
        source,
        pageTitle: titleStr
      })
    })
  })
}
