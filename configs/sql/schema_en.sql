-- English business database for the `db_query` tool (voice tool-calling, EN).
-- Purpose: a fixed, realistic schema so synthetic `db_query(question=...)` cases
-- are answerable/gradable, and so a later NL->SQL backend has a target. The
-- assistant model never touches this directly: db_query takes a natural-language
-- question and the backend translates it (read-only) against this schema.

CREATE TABLE IF NOT EXISTS customers (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    country      TEXT,
    segment      TEXT,            -- e.g. 'enterprise', 'smb', 'startup'
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS products (
    id           SERIAL PRIMARY KEY,
    name         TEXT NOT NULL,
    category     TEXT,
    unit_price   NUMERIC(10, 2) NOT NULL,
    in_stock     INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS employees (
    id           SERIAL PRIMARY KEY,
    full_name    TEXT NOT NULL,
    team         TEXT,
    title        TEXT,
    hired_at     DATE
);

CREATE TABLE IF NOT EXISTS orders (
    id           SERIAL PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    product_id   INTEGER NOT NULL REFERENCES products(id),
    quantity     INTEGER NOT NULL CHECK (quantity > 0),
    total_amount NUMERIC(12, 2) NOT NULL,
    status       TEXT NOT NULL DEFAULT 'open'
                 CHECK (status IN ('open', 'shipped', 'delivered', 'cancelled')),
    ordered_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders ((ordered_at::date));

CREATE TABLE IF NOT EXISTS meetings (
    id           SERIAL PRIMARY KEY,
    employee_id  INTEGER NOT NULL REFERENCES employees(id),
    customer_id  INTEGER REFERENCES customers(id),
    subject      TEXT,
    scheduled_at TIMESTAMPTZ NOT NULL
);

-- Read-only role for db_query's NL->SQL backend (defence in depth).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
        CREATE ROLE agent_ro LOGIN PASSWORD 'CHANGE_ME';  -- pragma: allowlist secret (placeholder)
    END IF;
END $$;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON customers, products, employees, orders, meetings TO agent_ro;

-- ---------------------------------------------------------------------------
-- Demo data
-- ---------------------------------------------------------------------------

INSERT INTO customers (name, country, segment) VALUES
    ('Acme Corp',      'US', 'enterprise'),
    ('Globex',         'UK', 'enterprise'),
    ('Initech',        'US', 'smb'),
    ('Hooli',          'US', 'startup'),
    ('Umbrella Ltd',   'DE', 'smb')
ON CONFLICT DO NOTHING;

INSERT INTO products (name, category, unit_price, in_stock) VALUES
    ('Widget Pro',     'hardware', 49.99, 1200),
    ('Widget Lite',    'hardware', 19.99, 0),
    ('Cloud Plan',     'service',  99.00, 9999),
    ('Support Gold',   'service', 499.00, 9999),
    ('Gadget X',       'hardware', 149.00, 30)
ON CONFLICT DO NOTHING;

INSERT INTO employees (full_name, team, title, hired_at) VALUES
    ('Alice Johnson',  'Sales',       'Account Executive', '2022-03-01'),
    ('Bob Smith',      'Engineering',  'Backend Engineer', '2021-09-15'),
    ('Carol Lee',      'Support',      'Support Lead',     '2023-01-10'),
    ('David Kim',      'Sales',        'Sales Manager',    '2020-06-20')
ON CONFLICT DO NOTHING;

INSERT INTO orders (customer_id, product_id, quantity, total_amount, status, ordered_at) VALUES
    (1, 1, 10,  499.90, 'shipped',   now() - INTERVAL '3 days'),
    (1, 3,  1,   99.00, 'open',      now() - INTERVAL '1 day'),
    (2, 5,  2,  298.00, 'delivered', now() - INTERVAL '10 days'),
    (3, 1,  5,  249.95, 'open',      now() - INTERVAL '2 days'),
    (4, 4,  1,  499.00, 'cancelled', now() - INTERVAL '5 days')
ON CONFLICT DO NOTHING;

INSERT INTO meetings (employee_id, customer_id, subject, scheduled_at) VALUES
    (1, 1, 'Quarterly review',   now() + INTERVAL '2 days'),
    (4, 2, 'Renewal discussion', now() + INTERVAL '5 days')
ON CONFLICT DO NOTHING;
