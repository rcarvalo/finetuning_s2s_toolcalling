-- Schéma PostgreSQL de l'agent d'accueil (Phase 3).
-- L'agent n'accède à la base qu'en LECTURE SEULE via le rôle agent_ro ;
-- les écritures (notifications, prise de RDV) passent par les services métier.

CREATE EXTENSION IF NOT EXISTS unaccent;

-- unaccent() n'est pas IMMUTABLE par défaut ; wrapper pour index/requêtes.
CREATE OR REPLACE FUNCTION immutable_unaccent(text)
RETURNS text AS $$ SELECT unaccent('unaccent', $1) $$
LANGUAGE sql IMMUTABLE PARALLEL SAFE STRICT;

CREATE TABLE IF NOT EXISTS employees (
    id              SERIAL PRIMARY KEY,
    full_name       TEXT NOT NULL,
    email           TEXT UNIQUE,
    phone           TEXT,
    team            TEXT,
    office_location TEXT
);

CREATE TABLE IF NOT EXISTS appointments (
    id            SERIAL PRIMARY KEY,
    visitor_name  TEXT NOT NULL,
    employee_id   INTEGER NOT NULL REFERENCES employees(id),
    scheduled_at  TIMESTAMPTZ NOT NULL,
    location      TEXT,
    status        TEXT NOT NULL DEFAULT 'confirmed'
                  CHECK (status IN ('confirmed', 'pending', 'cancelled', 'done')),
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_appointments_date ON appointments ((scheduled_at::date));
CREATE INDEX IF NOT EXISTS idx_appointments_visitor
    ON appointments (immutable_unaccent(lower(visitor_name)));

CREATE TABLE IF NOT EXISTS locations (
    id         SERIAL PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    floor      TEXT,
    directions TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS guest_wifi (
    id          SERIAL PRIMARY KEY,
    ssid        TEXT NOT NULL,
    password    TEXT NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL
);

-- Trace des notifications (alimentée par le service de notification, pas par l'agent).
CREATE TABLE IF NOT EXISTS notifications (
    id             SERIAL PRIMARY KEY,
    recipient_kind TEXT NOT NULL CHECK (recipient_kind IN ('employee', 'receptionist')),
    recipient      TEXT NOT NULL,
    message        TEXT NOT NULL,
    channel        TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    status         TEXT NOT NULL DEFAULT 'sent'
);

-- Rôle lecture seule pour l'outil query_database (défense en profondeur,
-- en plus de default_transaction_read_only et de la garde syntaxique).
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_ro') THEN
        CREATE ROLE agent_ro LOGIN PASSWORD 'CHANGE_ME';  -- pragma: allowlist secret (placeholder)
    END IF;
END $$;
-- GRANT CONNECT ON DATABASE reception TO agent_ro;  -- adapter au nom de la base
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON employees, appointments, locations, guest_wifi TO agent_ro;
-- volontairement PAS de SELECT sur notifications (peut contenir des données sensibles)

-- ---------------------------------------------------------------------------
-- Données de démonstration
-- ---------------------------------------------------------------------------

INSERT INTO employees (full_name, email, team, office_location) VALUES
    ('Claire Martin', 'claire.martin@example.com', 'Direction Produit', '2e étage, aile B'),
    ('Karim Benali',  'karim.benali@example.com',  'Ingénierie',        '3e étage, aile A'),
    ('Sophie Nguyen', 'sophie.nguyen@example.com', 'Ressources Humaines', '1er étage, accueil RH')
ON CONFLICT (email) DO NOTHING;

INSERT INTO appointments (visitor_name, employee_id, scheduled_at, location) VALUES
    ('Marie Dupont', 1, CURRENT_DATE + INTERVAL '14 hours', 'salle B2'),
    ('Jean Petit',   2, CURRENT_DATE + INTERVAL '10 hours 30 minutes', 'salle A1')
ON CONFLICT DO NOTHING;

INSERT INTO locations (name, floor, directions) VALUES
    ('salle B2',   '2e étage',        'Prenez l''ascenseur jusqu''au 2e étage, puis à droite : la salle B2 est la deuxième porte sur votre gauche.'),
    ('salle A1',   '3e étage',        'Montez au 3e étage, la salle A1 est face aux ascenseurs.'),
    ('cafétéria',  'rez-de-chaussée', 'Traversez le hall : la cafétéria est au fond à gauche, après les portiques.'),
    ('toilettes',  'rez-de-chaussée', 'Les toilettes sont dans le couloir à droite de l''accueil.')
ON CONFLICT (name) DO NOTHING;

INSERT INTO guest_wifi (ssid, password, valid_until) VALUES
    ('Entreprise-Guest', 'Bienvenue2026!', now() + INTERVAL '90 days');
