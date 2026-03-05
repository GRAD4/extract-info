"""
Neo4j data loader.

Loads the company-capability dataset into Neo4j as a property graph:

    (Company {id, name, city, province, lat, lon})
        -[:HAS_CAPABILITY]->
    (Capability {name, category})

    (Company)
        -[:CERTIFIED_WITH]->
    (Certification {name})

Nodes are merged (MERGE) so the script is idempotent — safe to re-run.
"""
from neo4j import GraphDatabase


def load(companies: list[dict], uri: str, user: str, password: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # Clear existing data for a clean demo run
        session.run("MATCH (n) DETACH DELETE n")
        print("[Neo4j] Cleared existing nodes.")

        for company in companies:
            # Upsert the Company node
            session.run(
                """
                MERGE (c:Company {id: $id})
                SET c.name     = $name,
                    c.city     = $city,
                    c.province = $province,
                    c.lat      = $lat,
                    c.lon      = $lon
                """,
                id=company["id"],
                name=company["name"],
                city=company["city"],
                province=company["province"],
                lat=company["lat"],
                lon=company["lon"],
            )

            # Upsert Capability nodes and relationships
            for cap in company.get("capabilities", []):
                session.run(
                    """
                    MERGE (cap:Capability {name: $name, category: $category})
                    WITH cap
                    MATCH (c:Company {id: $company_id})
                    MERGE (c)-[:HAS_CAPABILITY]->(cap)
                    """,
                    name=cap["name"],
                    category=cap["category"],
                    company_id=company["id"],
                )

            # Upsert Certification nodes and relationships
            for cert in company.get("certifications", []):
                session.run(
                    """
                    MERGE (cert:Certification {name: $name})
                    WITH cert
                    MATCH (c:Company {id: $company_id})
                    MERGE (c)-[:CERTIFIED_WITH]->(cert)
                    """,
                    name=cert,
                    company_id=company["id"],
                )

        result = session.run("MATCH (n) RETURN count(n) AS total")
        total = result.single()["total"]
        print(f"[Neo4j] Loaded {len(companies)} companies. Total nodes: {total}")

    driver.close()
