import csv
import random

with open("nivel_hemoglobina_train.csv", "w", newline="", encoding="utf-8") as archivo:

    writer = csv.writer(archivo)

    writer.writerow([
        "id_genero",
        "altitud",
        "edad",
        "valor_min",
        "valor_max",
        "valor_hemoglobina",
        "estado_diagnostico"
    ])

    for _ in range(300):

        genero = random.choice([1, 2])

        if genero == 1:
            vmin = 14.0
            vmax = 18.0
        else:
            vmin = 13.0
            vmax = 16.5

        edad = random.randint(18, 80)

        altitud = random.choice([
            400,
            2558,
            3640,
            3735,
            4060
        ])

        estado = random.choice([
            "Anemia",
            "Estable",
            "Poliglobulia"
        ])

        if estado == "Anemia":
            hb = round(
                random.uniform(vmin - 5, vmin - 0.1),
                2
            )

        elif estado == "Poliglobulia":
            hb = round(
                random.uniform(vmax + 0.1, vmax + 5),
                2
            )

        else:
            hb = round(
                random.uniform(vmin, vmax),
                2
            )

        writer.writerow([
            genero,
            altitud,
            edad,
            vmin,
            vmax,
            hb,
            estado
        ])

print("CSV generado correctamente.")
