-- ─────────────────────────────────────────────
-- FILE: 01_schema.sql
-- PURPOSE: Create all 4 tables in PostgreSQL
-- RUN THIS FIRST before any other SQL file
-- ─────────────────────────────────────────────

-- Drop tables if they already exist (clean slate)
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS hubs;
DROP TABLE IF EXISTS Skus;

-- Hubs table
CREATE TABLE Hubs (
    hub_id    VARCHAR(5)  PRIMARY KEY,
    hub_name  VARCHAR(50),
    city      VARCHAR(30),
    state     VARCHAR(30),
    tier      VARCHAR(10)
);

-- SKUs table
CREATE TABLE Skus (
    sku_id       VARCHAR(5)  PRIMARY KEY,
    product_name VARCHAR(60),
    category     VARCHAR(30),
    price        NUMERIC(8,2)
);

-- Users table
CREATE TABLE users (
    user_id            VARCHAR(10) PRIMARY KEY,
    hub_id             VARCHAR(5)  REFERENCES hubs(hub_id),
    registration_date  DATE,
    last_active_date   DATE,
    days_inactive      INTEGER,
    wallet_balance     NUMERIC(10,2),
    total_recharges    INTEGER,
    avg_recharge_value NUMERIC(8,2),
    customer_segment   VARCHAR(20),
    is_churned         SMALLINT,
    gender             CHAR(1),
    preferred_sku      VARCHAR(5),
    tenure_days        INTEGER
);

-- Transactions table
CREATE TABLE transactions (
    txn_id    VARCHAR(12) PRIMARY KEY,
    user_id   VARCHAR(10) ,
    hub_id    VARCHAR(5)  REFERENCES hubs(hub_id),
    sku_id    VARCHAR(5)  REFERENCES skus(sku_id),
    txn_type  VARCHAR(20),
    txn_date  DATE,
    txn_month VARCHAR(7),
    amount    NUMERIC(10,2),
    status    VARCHAR(15)
);

select * from Hubs;
select * from Skus;
select * from users;
select * from transactions;

