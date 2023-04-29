from autocorrect import Speller

corrector = Speller(lang="es")
palabra="Sanana"

if not corrector(palabra)==palabra:
    salida = corrector(palabra)
else:
    salida=palabra

print(salida)