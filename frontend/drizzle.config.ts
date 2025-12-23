import "dotenv/config"

export default {
    out: "./migrations",
    schema: "src/database/schema.ts",
    dialect: "postgresql",
    dbCredentials: {
        url: process.env.DATABASE_URL!
    }
}
