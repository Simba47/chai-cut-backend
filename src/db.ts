import postgres from 'postgres'

// onnotice: startup migrations use IF NOT EXISTS, and Postgres reports every skipped one as a NOTICE
const db = postgres(process.env.DATABASE_URL!, { ssl: 'require', onnotice: () => {} })
export default db
