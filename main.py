from main_trainer import execute
from predictor import predictor, sing_predictor

if __name__ == "__main__":
     
    # element = "Ce dernier volet de la trilogie a beau être le moins bon, il reste un très grand film qui fait du ""Parrain"" une des plus grandes sagas de l'histoire du cinéma. Michael Corleone a vieilli, vit avec sa culpabilité et veut se lancer dans des affaires légales. Mais quoiqu'il fasse sa vie semble être vouée à côtoyer la mort et même si son jeune neveu vient l'épauler, ce film prend des allures de tragédie jusqu'au final où comme chaque opus, un évènement vient marquer la fin d'une période à grand renfort de tueries. Le scénario est très bien écrit, s'adaptant parfaitement à l'époque du film et le casting est toujours aussi impeccable, si l'on regrettera l'absence de Robert Duvall, Al Pacino est toujours grandiose dans un de ses rôles les plus marquants, Andy Garcia est très convaincant, Talia Shire surprend avec son personnage de sœur qui décide de prendre les rênes et Eli Wallach est irrésistible en vieux mafioso. La musique et la mise en scène sont superbes."

    # element, pred = sing_predictor(element)

    # print(element)
    # print(pred)
    predictor()
    
    # print("Start training...")
    # execute()
    # print("Training completed!")
