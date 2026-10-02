from ber.normalize import norm_addr, norm_name, phon
from ber.blocking import block_tokens


def test_legal_forms_canonical():
    assert norm_name("Sri ONE TECHNOLOGIES PVT LTD")[1] == ["one", "technologies"]
    assert norm_name("One Technologies Private Limited")[2] == ["pvt", "ltd"]
    assert norm_name("L.L.P. Vivek India Infratech")[2] == ["llp"]


def test_web_names():
    canon, core, legal, web = norm_name("4220firstave.Com")
    assert core == ["4220firstave"] and web == 1
    assert norm_name("Gilda's Pharmaceuticals Ltd | www.gildasph.com")[1] == ["gildas", "pharmaceuticals"]


def test_address_abbreviations_and_numbers():
    a, toks, nums = norm_addr("ASHBURN, 0020718 ADAMS MILL PLACE, VA")
    assert "20718" in toks and "pl" in toks and nums == ["20718"]
    assert norm_addr("4809B Harrison Ferry Road, Hurlock, Maryland")[0] == "4809b harrison ferry rd hurlock md"
    assert norm_addr("N° 32 R DES LAURIERS, PORNIC")[1][:3] == ["n", "32", "rue"]


def test_phonetic_key_transliteration():
    for a, b in [("apex", "epeks"), ("properties", "puraapprttttiis"), ("first", "phrstt")]:
        assert phon(a) == phon(b)


def test_block_tokens_open_country_and_empty_address():
    t = block_tokens("classic foundation", "")
    assert "n:classic" in t and "a:" not in t
