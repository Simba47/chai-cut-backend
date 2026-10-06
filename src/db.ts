import postgres from 'postgres'

// LOCAL_EXPORT_DIR set (src/scripts/local-export.ts): an export on a developer PC reads the
// database but never changes it — the connection is read-only (Postgres refuses any write), and
// the job's writes are skipped (and logged) instead of failing the export
const localExport = !!process.env.LOCAL_EXPORT_DIR

// onnotice: startup migrations use IF NOT EXISTS, and Postgres reports every skipped one as a NOTICE
const sql = postgres(process.env.DATABASE_URL!, {
  ssl: 'require', onnotice: () => {},
  ...(localExport ? { connection: { default_transaction_read_only: true } } : {}),
})
const WRITE = /^\s*(update|insert|delete|alter|create|drop|truncate)\b/i
const db: typeof sql = localExport
  ? new Proxy(sql, {
      apply(target, thisArg, args) {
        const first = Array.isArray(args[0]) ? String(args[0][0]) : ''
        if (WRITE.test(first)) {
          console.log(`[local export] database write skipped: ${first.trim().split('\n')[0].slice(0, 70)}…`)
          return Promise.resolve([])
        }
        return Reflect.apply(target, thisArg, args)
      },
    })
  : sql
export default db
