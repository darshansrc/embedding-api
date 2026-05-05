// src/queue.ts

type Task<T> = () => Promise<T>;

interface QueueEntry<T> {
  task: Task<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
}

class AsyncQueue {
  private queue: Array<QueueEntry<any>> = [];
  private running = 0;
  private readonly concurrency: number;
  private readonly maxSize: number;

  constructor(concurrency = 1, maxSize = 50) {
    this.concurrency = concurrency;
    this.maxSize = maxSize;
  }

  /** Number of tasks waiting to run */
  get depth(): number {
    return this.queue.length;
  }

  /** Number of tasks currently running */
  get active(): number {
    return this.running;
  }

  add<T>(task: Task<T>): Promise<T> {
    if (this.queue.length >= this.maxSize) {
      return Promise.reject(
        new Error(`Queue full (max ${this.maxSize} pending requests)`)
      );
    }

    return new Promise<T>((resolve, reject) => {
      this.queue.push({ task, resolve, reject });
      this.run();
    });
  }

  private async run(): Promise<void> {
    if (this.running >= this.concurrency || this.queue.length === 0) return;

    this.running++;
    const entry = this.queue.shift()!;

    try {
      const result = await entry.task();
      entry.resolve(result);
    } catch (err) {
      entry.reject(err);
    } finally {
      this.running--;
      this.run();
    }
  }
}

// Single shared queue — serialises all embedding work (CPU-bound)
export const embedQueue = new AsyncQueue(1, 50);
