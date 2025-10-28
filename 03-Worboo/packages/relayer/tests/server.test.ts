import { mkdtempSync, rmSync } from 'fs'
import { join, resolve } from 'path'
import { tmpdir } from 'os'
import { describe, expect, it, afterAll } from 'vitest'

import { ProcessedEventStore } from '../src/store'
import { RelayerMetrics } from '../src/metrics'
import { startHealthServer } from '../src/server'
import { createLogger } from '../src/logger'

const setupStore = async () => {
  const dir = mkdtempSync(join(tmpdir(), 'worboo-health-server-'))
  const cachePath = resolve(dir, 'events.jsonl')
  const healthPath = resolve(dir, 'health.json')
  const store = await ProcessedEventStore.open({ filePath: cachePath })
  const metrics = new RelayerMetrics({ healthPath })
  return { dir, cachePath, healthPath, store, metrics }
}

describe('startHealthServer', () => {
  afterAll(() => {
    // cleanup tmp directories is handled per-test
  })

  it('serves health snapshots over HTTP', async () => {
    const { dir, store, metrics } = await setupStore()
    const logger = createLogger({
      infoWriter: () => undefined,
      errorWriter: () => undefined,
    })

    const server = await startHealthServer({
      store,
      metrics,
      host: '127.0.0.1',
      port: 0,
      logger,
    })

    const address = server.address()
    expect(address).not.toBeNull()

    const url = `http://${address?.address}:${address?.port}/healthz`
    const response = await fetch(url)
    expect(response.status).toBe(200)
    const payload = await response.json()
    expect(payload.status).toBeDefined()
    expect(payload.queueDepth).toBe(0)

    await server.close()
    rmSync(dir, { recursive: true, force: true })
  })

  it('returns 404 for unknown routes', async () => {
    const { dir, store, metrics } = await setupStore()
    const logger = createLogger({
      infoWriter: () => undefined,
      errorWriter: () => undefined,
    })

    const server = await startHealthServer({
      store,
      metrics,
      host: '127.0.0.1',
      port: 0,
      logger,
    })

    const address = server.address()
    const url = `http://${address?.address}:${address?.port}/unknown`
    const response = await fetch(url)
    expect(response.status).toBe(404)

    await server.close()
    rmSync(dir, { recursive: true, force: true })
  })
})
