# SIMPLE_QUERY = """
# SELECT ?s ?r ?o
# WHERE
# {
#     ?s ?r ?o .
# }
# """

SIMPLE_QUERY = """
SELECT DISTINCT ?s ?r ?o
WHERE
{
    {
        SELECT ?s ?r ?o
        WHERE
        {
            ?s ?r ?o .
        }
        GROUP BY ?s ?r
        HAVING(count(?o) = 1)
    }
    {
        SELECT ?s ?r ?o
        WHERE
        {
            ?s ?r ?o .
        }
        GROUP BY ?o ?r
        HAVING(count(?s) = 1)
    }
}
"""


SET_QUERY = """
SELECT ?s ?r ?o ?len
WHERE
{
    {
        SELECT ?s ?r (COUNT(?o1) as ?len)
        (GROUP_CONCAT(DISTINCT(STR(?o1));separator="|") AS ?o)
        WHERE
        {
            ?s ?r ?o1 .
        }
        GROUP BY ?s ?r
        HAVING(COUNT(?o1) > 1)
    }
    UNION
    {
        SELECT ?o ?r (COUNT(?s1) as ?len)
        (GROUP_CONCAT(DISTINCT(STR(?s1));separator="|") AS ?s)
        WHERE
        {
            ?s1 ?r ?o .
        }
        GROUP BY ?o ?r
        HAVING(COUNT(?s1) > 1)
    }
}
"""

COND_QUERY = """
SELECT *
WHERE
{
    {
        SELECT ?s ?r ?o ?r1 ?o1
        WHERE
        {
            ?s ?r ?o .
            ?s ?r1 ?o1 .
            FILTER(?o != ?o1)
        }
        GROUP BY ?o ?o1 ?r ?r1
        HAVING(COUNT(?s) = 1)
    }
    UNION
    {
        SELECT ?s ?r ?o ?r1 ?s1
        WHERE
        {
            ?s ?r ?o .
            ?s1 ?r1 ?o .
            FILTER(?s != ?s1)
        }
        GROUP BY ?s ?s1 ?r ?r1
        HAVING(COUNT(?o) = 1)
    }
    UNION
    {
        SELECT ?s ?r ?o ?r1 ?b
        WHERE
        {
            ?b ?r ?o .
            ?s ?r1 ?b .
            FILTER(?o != ?s)
        }
        GROUP BY ?s ?o ?r ?r1
        HAVING(COUNT(?b) = 1)
    }
    FILTER(?r != ?r1)
}
"""


MH_QUERY = """
SELECT *
WHERE
{
    {
        SELECT ?s ?r ?o ?r1 ?o1
        WHERE
        {
            ?s ?r ?o .
            ?s ?r1 ?o1 .
            FILTER(?o != ?o1)
        }
        GROUP BY ?s ?r ?r1
        HAVING(COUNT(?o) = 1)
    }
    UNION
    {
        SELECT ?s ?r ?o ?r1 ?s1
        WHERE
        {
            ?s ?r ?o .
            ?s1 ?r1 ?o .
            FILTER(?s != ?s1)
        }
        GROUP BY ?o ?r ?r1
        HAVING(COUNT(?s) = 1)
    }
    UNION
    {
        SELECT ?s ?r ?o ?r1 ?b
        WHERE
        {
            ?b ?r ?o .
            ?s ?r1 ?b .
            FILTER(?o != ?s)
        }
        GROUP BY ?b ?r ?r1
        HAVING(COUNT(?o) = 1)
    }
    FILTER(?r != ?r1)
}
"""

COMP_QUERY = """
SELECT ?s ?s1 ?r ?o ?o1
WHERE
{
    ?s ?r ?o .
    ?s1 ?r ?o1 .
    FILTER(?s != ?s1) .
    FILTER(strstarts(str(?o), "news://comp#")) .
    FILTER(strstarts(str(?o1), "news://comp#")) .
}
"""
