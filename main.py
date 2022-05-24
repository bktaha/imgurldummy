# MOKSHA

from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
BASE_S3_URL = 'https://mokshawd.s3.ap-south-1.amazonaws.com/'

class InputImageURL(BaseModel):
    imgur_url: str
    model_name: Optional[str] = 'sn'

class OutputListModel(BaseModel):
    img_features: List[float]

class OutputSimilarModel(BaseModel):
    img_rank: int
    img_reference: int
    img_url: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/features", response_model=OutputListModel)
def get_features(itmurl: InputImageURL):
    from turicreate import load_images, image_analysis
    image_sframe = load_images(itmurl.imgur_url)
    deep_features_sframe = image_analysis.get_deep_features(image_sframe['image'], model_name='squeezenet_v1.1')
    return OutputListModel(img_features= list(deep_features_sframe[0]))

def get_filenames(ref_id: List[int], model_name: str =  'rn'):
    from pandas import read_csv
    img_paths = read_csv('allimages_{}_paths.csv'.format(model_name))
    img_paths.columns = ['ref_id', 'img_filename']
    img_paths['img_filename'] = img_paths['img_filename'].str.replace(' ', '+', regex=False)
    return img_paths.loc[ref_id, 'img_filename']

@app.post("/similar", response_model=List[OutputSimilarModel])
def get_similar(itmurl: InputImageURL):
    from turicreate import load_images, load_model
    image_sframe = load_images(itmurl.imgur_url)
    sn_model = load_model('allimages_{}.model'.format(itmurl.model_name))

    sn_answers = sn_model.query(image_sframe)
    output_similar = [OutputSimilarModel(
            img_rank=sn_rank,
            img_reference=sn_ref,
            img_url=BASE_S3_URL+sn_filename)
        for sn_rank, sn_ref, sn_filename in zip(
            range(1,6),
            sn_answers["reference_label"],
            get_filenames(sn_answers["reference_label"], itmurl.model_name))]
    return output_similar
