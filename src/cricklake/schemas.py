"""Explicit data contracts.

The raw Cricsheet match schema is declared up front instead of inferred: inference over thousands of
deeply nested documents is slow and unstable (e.g. `season` is sometimes 2019 and sometimes "2019/20").
Unknown fields are ignored, missing ones become null, unparseable documents land in `_corrupt_record`.
"""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    IntegerType,
    MapType,
    StringType,
    StructField,
    StructType,
)


def _struct(**fields) -> StructType:
    return StructType([StructField(name, dtype) for name, dtype in fields.items()])


FIELDER = _struct(name=StringType(), substitute=BooleanType())
WICKET = _struct(player_out=StringType(), kind=StringType(), fielders=ArrayType(FIELDER))
DELIVERY = _struct(
    batter=StringType(),
    bowler=StringType(),
    non_striker=StringType(),
    runs=_struct(batter=IntegerType(), extras=IntegerType(), total=IntegerType(), non_boundary=BooleanType()),
    extras=_struct(wides=IntegerType(), noballs=IntegerType(), byes=IntegerType(), legbyes=IntegerType(),
                   penalty=IntegerType()),
    wickets=ArrayType(WICKET),
)
OVER = _struct(over=IntegerType(), deliveries=ArrayType(DELIVERY))
INNINGS = _struct(
    team=StringType(),
    overs=ArrayType(OVER),
    target=_struct(overs=DoubleType(), runs=IntegerType()),
    super_over=BooleanType(),
    declared=BooleanType(),
    forfeited=BooleanType(),
)
INFO = _struct(
    balls_per_over=IntegerType(),
    city=StringType(),
    dates=ArrayType(StringType()),
    event=_struct(name=StringType(), match_number=IntegerType(), group=StringType(), stage=StringType()),
    gender=StringType(),
    match_type=StringType(),
    outcome=_struct(
        winner=StringType(),
        by=_struct(runs=IntegerType(), wickets=IntegerType(), innings=IntegerType()),
        result=StringType(),
        method=StringType(),
        eliminator=StringType(),
    ),
    overs=IntegerType(),
    player_of_match=ArrayType(StringType()),
    players=MapType(StringType(), ArrayType(StringType())),
    registry=_struct(people=MapType(StringType(), StringType())),
    season=StringType(),
    team_type=StringType(),
    teams=ArrayType(StringType()),
    toss=_struct(decision=StringType(), winner=StringType()),
    venue=StringType(),
)
MATCH = StructType([
    StructField("match_id", StringType()),
    StructField("meta", _struct(data_version=StringType(), created=StringType(), revision=IntegerType())),
    StructField("info", INFO),
    StructField("innings", ArrayType(INNINGS)),
    StructField("_corrupt_record", StringType()),
])

# One ball event on the live feed (the streaming "topic").
BALL_EVENT = _struct(
    event_id=StringType(),
    event_ts=StringType(),
    match_id=StringType(),
    event_name=StringType(),
    season_year=IntegerType(),
    gender=StringType(),
    venue=StringType(),
    innings_no=IntegerType(),
    batting_team=StringType(),
    bowling_team=StringType(),
    over_no=IntegerType(),
    delivery_no=IntegerType(),
    batter=StringType(),
    bowler=StringType(),
    runs_batter=IntegerType(),
    runs_extras=IntegerType(),
    runs_total=IntegerType(),
    is_legal=BooleanType(),
    is_wicket=BooleanType(),
    wicket_kind=StringType(),
    player_out=StringType(),
    target_runs=IntegerType(),
    target_overs=DoubleType(),
)
