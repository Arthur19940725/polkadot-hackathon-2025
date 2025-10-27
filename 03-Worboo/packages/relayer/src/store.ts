import { appendFile, mkdir, readFile, writeFile } from 'fs/promises'
import { dirname, resolve } from 'path'
import { existsSync } from 'fs'

export type ProcessedEventMetadata = {
  txHash: string
  mintedAt?: number
}

export type ProcessedEventStoreOptions = {
  filePath?: string
}

const DEFAULT_CACHE_PATH = resolve(
  process.cwd(),
  '.cache',
  'processed-events.jsonl'
)

export class ProcessedEventStore {
  private readonly filePath: string
  private readonly processed = new Set<string>()

  private constructor(filePath: string) {
    this.filePath = filePath
  }

  static async open(
    options: ProcessedEventStoreOptions = {}
  ): Promise<ProcessedEventStore> {
    const filePath =
      options.filePath ?? process.env.RELAYER_CACHE_PATH ?? DEFAULT_CACHE_PATH
    const store = new ProcessedEventStore(filePath)
    await store.initialise()
    return store
  }

  get size(): number {
    return this.processed.size
  }

  get path(): string {
    return this.filePath
  }

  hasProcessed(key: string): boolean {
    return this.processed.has(key)
  }

  async markProcessed(
    key: string,
    meta: ProcessedEventMetadata
  ): Promise<void> {
    if (this.processed.has(key)) {
      return
    }

    this.processed.add(key)
    await this.appendRecord({
      key,
      txHash: meta.txHash,
      mintedAt: meta.mintedAt ?? Date.now(),
    })
  }

  private async initialise(): Promise<void> {
    await this.ensureDirectory()

    if (!existsSync(this.filePath)) {
      await writeFile(this.filePath, '', { encoding: 'utf-8' })
      return
    }

    const buffer = await readFile(this.filePath, { encoding: 'utf-8' })
    if (!buffer.trim()) {
      return
    }

    buffer.split('\n').forEach((line) => {
      const trimmed = line.trim()
      if (!trimmed) return
      try {
        const parsed = JSON.parse(trimmed) as { key?: string }
        if (parsed.key) {
          this.processed.add(parsed.key)
        }
      } catch {
        throw new Error(`Failed to parse processed event entry: ${trimmed}`)
      }
    })
  }

  private async ensureDirectory(): Promise<void> {
    const dir = dirname(this.filePath)
    if (!existsSync(dir)) {
      await mkdir(dir, { recursive: true })
    }
  }

  private async appendRecord(record: {
    key: string
    txHash: string
    mintedAt: number
  }): Promise<void> {
    await appendFile(this.filePath, `${JSON.stringify(record)}\n`, {
      encoding: 'utf-8',
    })
  }
}
