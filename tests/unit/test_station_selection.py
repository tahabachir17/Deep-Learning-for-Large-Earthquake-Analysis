from src.data.station_selection import draw_combinations


def test_draw_combinations_respects_cap():
    combos = draw_combinations(["A", "B", "C", "D"], nst=2, max_combinations=3)
    assert len(combos) == 3
