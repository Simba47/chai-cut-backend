import postgres from 'postgres'

const db = postgres(process.env.DATABASE_URL!, { ssl: 'require' })
export default db
